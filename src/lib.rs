#![warn(missing_docs)]
#![doc(issue_tracker_base_url = "https://gitlab.101100.ca/ben1jen/playback-rs/-/issues")]
#![doc = include_str!("../docs.md")]

use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use color_eyre::eyre::{Report, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FrameCount, Sample, SampleFormat, SupportedBufferSize, SupportedStreamConfigRange};
use log::{debug, error, info, warn};
use samplerate::{ConverterType, Samplerate};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::default;

pub use symphonia::core::probe::Hint;

#[derive(Debug, Clone, Copy, PartialEq)]
struct SongRequest {
	samples: usize,
	speed: f32,
}

#[derive(Debug)]
struct DecodingSong {
	requests_channel: SyncSender<SongRequest>,
	samples_channel: Mutex<Receiver<Option<Vec<f32>>>>,
	done: bool,
	target_buffer_size: usize,
	buffer: VecDeque<f32>,
	len: usize,
	playback_speed: f32,
}

impl DecodingSong {
	fn new(
		song: &Song,
		sample_rate: u32,
		channel_count: usize,
		target_buffer_size: usize,
		playback_speed: f32,
	) -> Result<DecodingSong> {
		// Reorder samples, this is really fast and gets us ownership over the song
		let samples = {
			let sample_count = song.samples[0].len();
			let mut samples = vec![0.0; channel_count * sample_count];
			for chan in 0..channel_count {
				if chan < 2 || chan < song.samples.len() {
					for sample in 0..sample_count {
						samples[sample * channel_count as usize + chan] =
							song.samples[chan % song.samples.len()][sample]
					}
				};
			}
			samples
		};
		let len = samples.len() * sample_rate as usize / song.sample_rate as usize;
		let (rtx, rrx) = mpsc::sync_channel::<SongRequest>(10);
		let (stx, srx) = mpsc::channel();
		let source_sample_rate = song.sample_rate;
		let (etx, erx) = mpsc::channel();
		thread::spawn(move || {
			let mut converter = match Samplerate::new(
				ConverterType::SincBestQuality,
				source_sample_rate,
				sample_rate,
				channel_count,
			) {
				Ok(converter) => {
					etx.send(Ok(())).unwrap();
					converter
				}
				Err(e) => {
					etx.send(Err(e)).unwrap();
					return;
				}
			};
			let mut current_sample = 0;
			let mut last_target_rate = sample_rate;
			while current_sample <= samples.len() {
				let request = rrx.recv().unwrap();
				let target_rate = (sample_rate as f32 / request.speed) as u32;
				let (min_target_rate, max_target_rate) =
					(source_sample_rate / 256 + 1, source_sample_rate * 256 - 1);
				if !(min_target_rate..=max_target_rate).contains(&target_rate) {
					warn!("Can't achieve correct sample rate conversion for requested sample rate, expect problems!");
				}
				let target_rate = target_rate.clamp(min_target_rate, max_target_rate);
				if target_rate != last_target_rate {
					converter.set_to_rate(target_rate);
				}
				let last_sample = (current_sample
					+ (request.samples + 1) * (source_sample_rate as usize)
						/ (target_rate.max(last_target_rate) as usize)
						/ channel_count * channel_count)
					.min(samples.len());
				let samples_input = &samples[current_sample..last_sample];
				// trace!(
				// 	"Got request for {} samples at a playback speed of {}. Using a target sample rate of {target_rate}, changed from {last_target_rate}. Sending {} samples to resampler as input.",
				// 	request.samples, request.speed, samples_input.len()
				// );
				last_target_rate = target_rate;
				current_sample = last_sample;
				let processed_samples = match if last_sample == samples.len() {
					converter.process_last(samples_input)
				} else {
					converter.process(samples_input)
				} {
					Ok(samples) => samples,
					Err(e) => {
						error!("Error converting sample rate: {e}");
						vec![0.0; request.samples * channel_count]
					}
				};
				// trace!("Sending {} samples over channel.", processed_samples.len());
				// Dropping the other end of the channel will cause this to error, which will stop decoding.
				if stx.send(Some(processed_samples)).is_err() {
					debug!("Ending resampling thread.");
					break;
				}
			}
		});
		erx.recv()??;
		rtx.send(SongRequest {
			samples: target_buffer_size,
			speed: playback_speed,
		})?;
		Ok(DecodingSong {
			requests_channel: rtx,
			samples_channel: Mutex::new(srx),
			done: false,
			target_buffer_size,
			buffer: VecDeque::new(),
			len,
			playback_speed,
		})
	}
	fn read_samples(&mut self, pos: usize, count: usize) -> (Vec<f32>, bool) {
		// TODO: Fix seeking properly.
		self.target_buffer_size = self.target_buffer_size.max(count * 2);
		let channel = self.samples_channel.lock().unwrap();
		if self.target_buffer_size + count > self.buffer.len() {
			self.requests_channel
				.send(SongRequest {
					samples: self.target_buffer_size + count - self.buffer.len(),
					speed: self.playback_speed,
				})
				.unwrap(); // This shouldn't be able to fail unless the thread stops which shouldn't be able to happen.
		}
		if !self.done {
			// Fetch samples until there are none left to fetch and we have enough.
			let mut sent_warning = false;
			loop {
				let got = channel.try_recv();
				match got {
					Ok(Some(buf)) => {
						self.buffer.append(&mut VecDeque::from(buf));
					}
					Ok(None) | Err(TryRecvError::Disconnected) => {
						self.done = true;
						break;
					}
					Err(TryRecvError::Empty) => {
						if self.buffer.len() > count {
							break;
						} else if !sent_warning {
							warn!("Waiting on resampler, this could cause audio choppyness. If you are a developer and this happens repeatedly in release mode please file an issue on playback-rs or message the maintainer (BEN1JEN#8140) on discord.");
							sent_warning = true;
						}
					}
				}
			}
		}
		let mut vec = Vec::new();
		let mut done = false;
		for _i in 0..count {
			if let Some(sample) = self.buffer.pop_front() {
				vec.push(sample);
			} else {
				done = true;
				break;
			}
		}
		(vec, done)
	}
	fn len(&self) -> usize {
		self.len
	}
}

type PlaybackState = (DecodingSong, usize);

#[derive(Clone)]
struct PlayerState {
	playback: Arc<RwLock<Option<PlaybackState>>>,
	next_samples: Arc<RwLock<Option<DecodingSong>>>,
	playing: Arc<RwLock<bool>>,
	playback_speed: Arc<RwLock<f32>>,
	sample_rate: u32,
	channel_count: usize,
	buffer_size: u32,
}

impl PlayerState {
	fn new(channel_count: u32, sample_rate: u32, buffer_size: FrameCount) -> Result<PlayerState> {
		Ok(PlayerState {
			playback: Arc::new(RwLock::new(None)),
			next_samples: Arc::new(RwLock::new(None)),
			playing: Arc::new(RwLock::new(true)),
			channel_count: channel_count as usize,
			sample_rate,
			buffer_size,
			playback_speed: Arc::new(RwLock::new(1.0)),
		})
	}
	fn write_samples<T: Sample>(&self, data: &mut [T]) {
		for sample in data.iter_mut() {
			*sample = Sample::from(&0.0);
		}
		if *self.playing.read().unwrap() {
			let mut playback = self.playback.write().unwrap();
			if playback.is_none() {
				if let Some(new_samples) = self.next_samples.write().unwrap().take() {
					*playback = Some((new_samples, 0));
				}
			}
			let mut done = false;
			if let Some((decoding_song, pos)) = playback.as_mut() {
				let mut neg_offset = 0;
				let data_len = data.len();
				let (mut samples, mut is_final) = decoding_song.read_samples(*pos, data_len);
				for (i, sample) in data.iter_mut().enumerate() {
					if i >= samples.len() {
						if let Some(new_samples) = self.next_samples.write().unwrap().take() {
							*decoding_song = new_samples;
							neg_offset = i;
							*pos = 0;
							(samples, is_final) =
								decoding_song.read_samples(*pos, data_len - neg_offset);
						} else {
							break;
						}
					}
					*sample = Sample::from(&samples[i - neg_offset]);
				}
				*pos += data_len - neg_offset;
				done = is_final;
			}
			if done {
				*playback = None;
			}
		}
	}
	fn decode_song(&self, song: &Song) -> Result<DecodingSong> {
		DecodingSong::new(
			song,
			self.sample_rate,
			self.channel_count,
			self.buffer_size as usize,
			*self.playback_speed.read().unwrap(),
		)
	}
	fn set_playback_speed(&self, speed: f32) {
		*self.playback_speed.write().unwrap() = speed;
		// TODO: This probably could be made better.
		if let Some(pb) = &mut *self.playback.write().unwrap() {
			pb.0.playback_speed = speed;
		}
		if let Some(samples) = &mut *self.next_samples.write().unwrap() {
			samples.playback_speed = speed;
		}
	}
	fn stop(&self) {
		*self.next_samples.write().unwrap() = None;
		*self.playback.write().unwrap() = None;
	}
	fn skip(&self) {
		*self.playback.write().unwrap() = None;
	}
	fn play_song(&self, song: &Song) -> Result<()> {
		let samples = self.decode_song(song)?;
		*self.next_samples.write().unwrap() = Some(samples);
		Ok(())
	}
	fn set_playing(&self, playing: bool) {
		*self.playing.write().unwrap() = playing;
	}
	fn get_position(&self) -> Option<(usize, usize)> {
		self.playback
			.read()
			.unwrap()
			.as_ref()
			.map(|(samples, pos)| (*pos, samples.len()))
	}
	fn seek(&self, position: usize) -> bool {
		if let Some((_samples, pos)) = self.playback.write().unwrap().as_mut() {
			*pos = position;
			true
		} else {
			false
		}
	}
	fn force_remove_next_song(&self) {
		let (mut playback, mut next_song) = (
			self.playback.write().unwrap(),
			self.next_samples.write().unwrap(),
		);
		if next_song.is_some() {
			*next_song = None;
		} else {
			*playback = None;
		}
	}
}

/// Manages playback of [Song]s through [cpal] and sample conversion through [samplerate].
pub struct Player {
	_stream: Box<dyn StreamTrait>,
	player_state: PlayerState,
}

impl Player {
	/// Creates a new [Player] to play [Song]s
	///
	/// On Linux, this prefers `pipewire`, `jack`, and `pulseaudio` devices over `alsa`.
	pub fn new() -> Result<Player> {
		let device = {
			let mut selected_host = cpal::default_host();
			for host in cpal::available_hosts() {
				if host.name().to_lowercase().contains("jack") {
					selected_host = cpal::host_from_id(host)?;
				}
			}
			info!("Selected Host: {:?}", selected_host.id());
			let mut selected_device = selected_host
				.default_output_device()
				.ok_or_else(|| Report::msg("No output device found."))?;
			for device in selected_host.output_devices()? {
				if let Ok(name) = device.name().map(|s| s.to_lowercase()) {
					if name.contains("pipewire") || name.contains("pulse") || name.contains("jack")
					{
						selected_device = device;
					}
				}
			}
			info!(
				"Selected Device: {}",
				selected_device
					.name()
					.unwrap_or_else(|_| "Unknown".to_string())
			);
			selected_device
		};
		let mut supported_configs = device.supported_output_configs()?.collect::<Vec<_>>();
		fn rank_supported_config(config: &SupportedStreamConfigRange) -> u32 {
			let chans = config.channels() as u32;
			let channel_rank = match chans {
				0 => 0,
				1 => 1,
				2 => 4,
				4 => 3,
				_ => 2,
			};
			let min_sample_rank = if config.min_sample_rate().0 <= 48000 {
				3
			} else {
				0
			};
			let max_sample_rank = if config.max_sample_rate().0 >= 48000 {
				3
			} else {
				0
			};
			let sample_format_rank = if config.sample_format() == SampleFormat::F32 {
				4
			} else {
				0
			};
			channel_rank + min_sample_rank + max_sample_rank + sample_format_rank
		}
		supported_configs.sort_by_key(|c_2| std::cmp::Reverse(rank_supported_config(c_2)));

		let supported_config = supported_configs
			.into_iter()
			.next()
			.ok_or_else(|| Report::msg("No supported output config."))?;

		let sample_rate_range =
			supported_config.min_sample_rate().0..supported_config.max_sample_rate().0;
		let supported_config = if sample_rate_range.contains(&48000) {
			supported_config.with_sample_rate(cpal::SampleRate(48000))
		} else if sample_rate_range.contains(&44100) {
			supported_config.with_sample_rate(cpal::SampleRate(44100))
		} else if sample_rate_range.end <= 48000 {
			supported_config.with_sample_rate(cpal::SampleRate(sample_rate_range.end))
		} else {
			supported_config.with_sample_rate(cpal::SampleRate(sample_rate_range.start))
		};
		let sample_format = supported_config.sample_format();
		let sample_rate = supported_config.sample_rate().0;
		let channel_count = supported_config.channels();
		let buffer_size = match supported_config.buffer_size() {
			SupportedBufferSize::Range { min, .. } => (*min).max(1024) * 2,
			SupportedBufferSize::Unknown => 1024 * 2,
		};
		let config = supported_config.into();
		let err_fn = |err| error!("A playback error has occured! {}", err);
		let player_state = PlayerState::new(channel_count as u32, sample_rate, buffer_size)?;
		info!(
			"SR, CC, SF: {}, {}, {:?}",
			sample_rate, channel_count, sample_format
		);
		let stream = {
			let player_state = player_state.clone();
			match sample_format {
				SampleFormat::F32 => device.build_output_stream(
					&config,
					move |data, _| player_state.write_samples::<f32>(data),
					err_fn,
				)?,
				SampleFormat::I16 => device.build_output_stream(
					&config,
					move |data, _| player_state.write_samples::<i16>(data),
					err_fn,
				)?,
				SampleFormat::U16 => device.build_output_stream(
					&config,
					move |data, _| player_state.write_samples::<u16>(data),
					err_fn,
				)?,
			}
		};
		// Not all platforms (*cough cough* windows *cough*) automatically run the stream upon creation, so do that here.
		stream.play()?;
		Ok(Player {
			_stream: Box::new(stream),
			player_state,
		})
	}
	/// Set the playback speed (This will also affect song pitch)
	pub fn set_playback_speed(&self, speed: f32) {
		self.player_state.set_playback_speed(speed);
	}
	/// Set the song that will play after the current song is over (or immediately if no song is currently playing)
	pub fn play_song_next(&self, song: &Song) -> Result<()> {
		self.player_state.play_song(song)
	}
	/// Start playing a song immediately, while discarding any song that might have been queued to play next.
	pub fn play_song_now(&self, song: &Song) -> Result<()> {
		self.player_state.stop();
		self.player_state.play_song(song)?;
		Ok(())
	}
	/// Used to replace the next song, or the current song if there is no next song.
	///
	/// This will remove the current song if no next song exists to avoid a race condition in case the current song ends after you have determined that the next song must be replaced but before you call this function.
	/// See also [`force_remove_next_song`](Player::force_remove_next_song)
	pub fn force_replace_next_song(&self, song: &Song) -> Result<()> {
		self.player_state.force_remove_next_song();
		self.player_state.play_song(song)?;
		Ok(())
	}
	/// Used to remove the next song, or the current song if there is no next song.
	///
	/// This will remove the current song if no next song exists to avoid a race condition in case the current song ends after you have determined that the next song must be replaced but before you call this function.
	/// See also [`force_replace_next_song`](Player::force_replace_next_song)
	pub fn force_remove_next_song(&self) -> Result<()> {
		self.player_state.force_remove_next_song();
		Ok(())
	}
	/// Stop playing any songs and remove a next song if it has been queued.
	///
	/// Note that this does not pause playback (use [`set_playing`](Player::set_playing)), meaning new songs will play upon adding them.
	pub fn stop(&self) {
		self.player_state.stop();
	}
	/// Skip the currently playing song (i.e. stop playing it immediately.
	///
	/// This will immediately start playing the next song if it exists.
	pub fn skip(&self) {
		self.player_state.skip();
	}
	fn get_duration_per_sample(&self) -> Duration {
		Duration::from_nanos(
			1000000000
				/ (self.player_state.sample_rate as u64 * self.player_state.channel_count as u64),
		)
	}
	/// Return the current playback position, if there is currently a song playing (see [`has_current_song`](Player::has_current_song))
	///
	/// See also [`seek`](Player::seek)
	pub fn get_playback_position(&self) -> Option<(Duration, Duration)> {
		self.player_state.get_position().map(|(current, total)| {
			let duration_per_sample = self.get_duration_per_sample();
			(
				duration_per_sample * current as u32,
				duration_per_sample * total as u32,
			)
		})
	}
	/// Set the current playback position if there is a song playing
	///
	/// Returns whether the seek was successful (whether there was a song to seek).
	/// Note that seeking past the end of the song will be successful and will cause playback to begin at the _beginning_ of the next song.
	///
	/// See also [`get_playback_position`](Player::get_playback_position)
	pub fn seek(&self, time: Duration) -> bool {
		let duration_per_sample = self.get_duration_per_sample();
		let samples = (time.as_nanos() / duration_per_sample.as_nanos()) as usize;
		self.player_state.seek(samples)
	}
	/// Sets whether playback is enabled or not, without touching the song queue.
	///
	/// See also [`is_playing`](Player::is_playing)
	pub fn set_playing(&self, playing: bool) {
		self.player_state.set_playing(playing);
	}
	/// Returns whether playback is currently paused
	///
	/// See also [`set_playing`](Player::set_playing)
	pub fn is_playing(&self) -> bool {
		*self.player_state.playing.read().unwrap()
	}
	/// Returns whether there is a song queued to play next after the current song has finished
	///
	/// If you want to check whether there is currently a song playing, use [`has_current_song`][Player::has_current_song] and [`is_playing`][Player::is_playing].
	/// This should always be queried before calling [`play_song_next`](Player::play_song_next) if you do not intend on replacing the song currently in the queue.
	pub fn has_next_song(&self) -> bool {
		self.player_state
			.next_samples
			.read()
			.expect("Next song mutex poisoned.")
			.is_some()
	}
	/// Returns whether there is a song currently playing (or about to start playing next audio frame)
	///
	/// Note that this **does not** indicate whether the current song is actively being played or paused, for that functionality you can use [is_playing](Self::is_playing).
	pub fn has_current_song(&self) -> bool {
		self.player_state
			.playback
			.read()
			.expect("Current song mutex poisoned.")
			.is_some() || self
			.player_state
			.next_samples
			.read()
			.expect("Next song mutex poisoned.")
			.is_some()
	}
}

/// Represents a single song that has been decoded into memory, can be played in a <Player> struct.
#[derive(Debug, Clone)]
pub struct Song {
	samples: Vec<Vec<f32>>,
	sample_rate: u32,
	channel_count: u32,
}

impl Song {
	/// Creates a new song using a reader of some kind and a type hint (the Symphonia hint type has been reexported at the crate root for convenience).
	pub fn new(reader: Box<dyn MediaSource>, hint: &Hint) -> Result<Song> {
		let media_source_stream =
			MediaSourceStream::new(reader, MediaSourceStreamOptions::default());
		let mut probe_result = default::get_probe().format(
			hint,
			media_source_stream,
			&FormatOptions {
				enable_gapless: true,
				..FormatOptions::default()
			},
			&MetadataOptions::default(),
		)?;
		let mut decoder = default::get_codecs().make(
			&probe_result
				.format
				.default_track()
				.ok_or_else(|| Report::msg("No default track in media file."))?
				.codec_params,
			&DecoderOptions::default(),
		)?;
		let mut song: Option<Song> = None;
		loop {
			match probe_result.format.next_packet() {
				Ok(packet) => {
					let decoded = decoder.decode(&packet)?;
					let spec = *decoded.spec();
					let song = if let Some(old_song) = &mut song {
						if spec.rate != old_song.sample_rate
							|| spec.channels.count() as u32 != old_song.channel_count
						{
							return Err(Report::msg("Sample rate or channel count of decoded does not match previous sample rate."));
						}
						old_song
					} else {
						song = Some(Song {
							samples: vec![Vec::new(); spec.channels.count()],
							sample_rate: spec.rate,
							channel_count: spec.channels.count() as u32,
						});
						song.as_mut().unwrap()
					};
					if decoded.frames() > 0 {
						let mut samples = SampleBuffer::new(decoded.frames() as u64, spec);
						samples.copy_interleaved_ref(decoded);
						for frame in samples.samples().chunks(spec.channels.count()) {
							for (chan, sample) in frame.iter().enumerate() {
								song.samples[chan].push(*sample)
							}
						}
					} else {
						warn!("Empty packet encountered while loading song!");
					}
				}
				Err(SymphoniaError::IoError(_)) => break,
				Err(e) => return Err(e.into()),
			}
		}
		song.ok_or_else(|| Report::msg("No song data decoded."))
	}
	/// Creates a [Song] by reading data from a file and using the file's extension as a format type hint.
	pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Song> {
		let mut hint = Hint::new();
		if let Some(extension) = path.as_ref().extension().and_then(|s| s.to_str()) {
			hint.with_extension(extension);
		}
		Self::new(Box::new(std::fs::File::open(path)?), &hint)
	}
}
