use std::sync::{Arc, RwLock};

use color_eyre::eyre::{Report, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use log::{error, info};
use samplerate::{ConverterType, Samplerate};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default;

type PlaybackState = (Vec<f32>, usize);

#[derive(Clone)]
struct PlayerState {
	playback: Arc<RwLock<Option<PlaybackState>>>,
	next_samples: Arc<RwLock<Option<Vec<f32>>>>,
	playing: Arc<RwLock<bool>>,
	sample_rate: u32,
	channel_count: usize,
}

impl PlayerState {
	fn new(channel_count: u32, sample_rate: u32) -> Result<PlayerState> {
		Ok(PlayerState {
			playback: Arc::new(RwLock::new(None)),
			next_samples: Arc::new(RwLock::new(None)),
			playing: Arc::new(RwLock::new(true)),
			channel_count: channel_count as usize,
			sample_rate,
		})
	}
	fn write_samples<T: Sample>(&self, data: &mut [T]) {
		for sample in data.iter_mut() {
			*sample = Sample::from(&0.0);
		}
		if *self.playing.read().unwrap() {
			let mut playback = self.playback.write().unwrap();
			let mut done = false;
			if playback.is_none() {
				if let Some(new_samples) = self.next_samples.write().unwrap().take() {
					*playback = Some((new_samples, 0));
				}
			}
			if let Some((samples, pos)) = playback.as_mut() {
				let mut neg_offset = 0;
				for (i, sample) in data.iter_mut().enumerate() {
					if *pos + i >= samples.len() {
						if let Some(new_samples) = self.next_samples.write().unwrap().take() {
							*samples = new_samples;
							neg_offset = i;
							*pos = 0;
						} else {
							done = true;
						}
					}
					if *pos + i < samples.len() {
						*sample = Sample::from(&samples[*pos + i]);
					}
				}
				*pos += data.len() - neg_offset;
			}
			if done {
				*playback = None;
			}
		}
	}
	fn decode_song(&self, song: &Song) -> Result<Vec<f32>> {
		let converter = Samplerate::new(
			ConverterType::SincFastest,
			song.sample_rate,
			self.sample_rate,
			self.channel_count,
		)?;
		let samples = {
			let sample_count = song.samples[0].len();
			let mut samples = vec![0.0; self.channel_count * sample_count];
			for chan in 0..self.channel_count {
				if chan < 2 || chan < song.samples.len() {
					for sample in 0..sample_count {
						samples[sample * self.channel_count as usize + chan] =
							song.samples[chan % song.samples.len()][sample]
					}
				};
			}
			samples
		};
		let t = std::time::Instant::now();
		let processed_samples = converter.process_last(&samples)?;
		info!("Converted song sample rate in {:?}", t.elapsed());
		Ok(processed_samples)
	}
	fn play_song(&self, song: &Song) -> Result<()> {
		let samples = self.decode_song(song)?;
		*self
			.playback
			.write()
			.map_err(|_err| Report::msg("Playback mutex poisoned."))? = Some((samples, 0));
		Ok(())
	}
	fn next_song(&self, song: &Song) -> Result<()> {
		let samples = self.decode_song(song)?;
		*self
			.next_samples
			.write()
			.map_err(|_err| Report::msg("Playing mutex poisoned."))? = Some(samples);
		Ok(())
	}
	fn set_playing(&mut self, playing: bool) -> Result<()> {
		*self
			.playing
			.write()
			.map_err(|_err| Report::msg("Playing mutex poisoned."))? = playing;
		Ok(())
	}
}

pub struct Player {
	_stream: Box<dyn StreamTrait>,
	player_state: PlayerState,
}

impl Player {
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
		let supported_config = device
			.supported_output_configs()?
			.next()
			.ok_or_else(|| Report::msg("No supported output config."))?
			.with_max_sample_rate();
		let sample_format = supported_config.sample_format();
		let sample_rate = supported_config.sample_rate().0;
		let channel_count = supported_config.channels();
		let config = supported_config.into();
		let err_fn = |err| error!("A playback error has occured! {}", err);
		let player_state = PlayerState::new(channel_count as u32, sample_rate)?;
		info!("SR, CC: {}, {}", sample_rate, channel_count);
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
		Ok(Player {
			_stream: Box::new(stream),
			player_state,
		})
	}
	pub fn play_song(&self, song: &Song) -> Result<()> {
		self.player_state.next_song(song)
	}
	pub fn has_next_song(&self) -> bool {
		self.player_state
			.next_samples
			.read()
			.expect("Next song mutex poisoned.")
			.is_some()
	}
	pub fn has_song(&self) -> bool {
		self.player_state
			.playback
			.read()
			.expect("Next song mutex poisoned.")
			.is_some()
	}
}

#[derive(Debug, Clone)]
pub struct Song {
	samples: Vec<Vec<f32>>,
	sample_rate: u32,
	channel_count: u32,
}

impl Song {
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
					let mut samples = SampleBuffer::new(decoded.frames() as u64, spec);
					samples.copy_interleaved_ref(decoded);
					for frame in samples.samples().chunks(spec.channels.count()) {
						for (chan, sample) in frame.iter().enumerate() {
							song.samples[chan].push(*sample)
						}
					}
				}
				Err(SymphoniaError::IoError(_)) => break,
				Err(e) => return Err(e.into()),
			}
		}
		song.ok_or_else(|| Report::msg("No song data decoded."))
	}
	pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Song> {
		let mut hint = Hint::new();
		if let Some(extension) = path.as_ref().extension().and_then(|s| s.to_str()) {
			hint.with_extension(extension);
		}
		Self::new(Box::new(std::fs::File::open(path)?), &hint)
	}
}

#[cfg(test)]
mod tests {
	#[test]
	fn it_works() {}
}
