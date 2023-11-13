// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use opencv;

mod cv;

fn main() -> opencv::Result<()> {
    let img = opencv::imgcodecs::imread_def("./img/list_2.jpg")?;
    let img = cv::imgproc_pipeline(img)?;
    let _ = opencv::imgcodecs::imwrite_def("./img/output.png", &img)?;

    // tauri::Builder::default()
    //     .run(tauri::generate_context!())
    //     .expect("error while running tauri application");

    Ok(())
}