use color_eyre::eyre::Result;

use playback_rs::{Player, Song};

fn main() -> Result<()> {
	color_eyre::install()?;

	let filenames = std::env::args().skip(1);
	let player = Player::new()?;
	for next_song in filenames {
		println!("Loading song '{}'...", next_song);
		let song = Song::from_file(&next_song)?;
		while player.has_next_song() {
			std::thread::sleep(std::time::Duration::from_millis(100));
		}
		println!("Queueing next song '{}'...", next_song);
		player.play_song_next(&song)?;
	}
	println!("Waiting for songs to finish.");
	while player.has_current_song() {
		std::thread::sleep(std::time::Duration::from_millis(100));
	}
	println!("Exiting.");

	Ok(())
}
