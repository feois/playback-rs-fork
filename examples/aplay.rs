use color_eyre::eyre::{Report, Result};

use playback_rs::{Player, Song};

fn main() -> Result<()> {
	color_eyre::install()?;

	let mut filenames = std::env::args().skip(1);
	let player = Player::new()?;
	while let Some(next_song) = filenames.next() {
		println!("Loading song '{}'...", next_song);
		let song = Song::from_file(&next_song)?;
		while player.has_next_song() {
			std::thread::sleep(std::time::Duration::from_millis(100));
		}
		println!("Queueing next song '{}'...", next_song);
		player.play_song(&song)?;
	}
	while player.has_song() {
		std::thread::sleep(std::time::Duration::from_millis(100));
	}
	println!("Exiting.");

	Ok(())
}
