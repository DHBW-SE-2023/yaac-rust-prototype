use std::{self, sync::Arc};

use average;
use itertools::{Either, Itertools};
use rayon::prelude::*;

use opencv::{
    self,
    core::{Mat, Point, Point2f, Size},
    prelude::MatTraitConst,
    text::BaseOCRTrait,
};

#[derive(Debug)]
struct Poly(opencv::core::Vector<Point>);

pub struct Rect {
    top_left: Point,
    top_right: Point,
    bottom_left: Point,
    bottom_right: Point,
}

impl Rect {
    pub fn top(&self) -> Line {
        Line {
            p1: self.top_left,
            p2: self.top_right,
        }
    }

    pub fn right(&self) -> Line {
        Line {
            p1: self.top_right,
            p2: self.bottom_right,
        }
    }

    pub fn bottom(&self) -> Line {
        Line {
            p1: self.bottom_right,
            p2: self.bottom_left,
        }
    }

    pub fn left(&self) -> Line {
        Line {
            p1: self.bottom_left,
            p2: self.top_left,
        }
    }
}

impl Into<opencv::core::Vector<Point>> for Rect {
    fn into(self) -> opencv::core::Vector<Point> {
        vec![
            self.top_right,
            self.top_left,
            self.bottom_left,
            self.bottom_right,
        ]
        .into()
    }
}

impl Into<opencv::core::Vector<Point2f>> for Rect {
    fn into(self) -> opencv::core::Vector<Point2f> {
        vec![
            Point2f {
                x: self.top_right.x as f32,
                y: self.top_right.y as f32,
            },
            Point2f {
                x: self.top_left.x as f32,
                y: self.top_left.y as f32,
            },
            Point2f {
                x: self.bottom_left.x as f32,
                y: self.bottom_left.y as f32,
            },
            Point2f {
                x: self.bottom_right.x as f32,
                y: self.bottom_right.y as f32,
            },
        ]
        .into()
    }
}

pub struct Line {
    pub p1: Point,
    pub p2: Point,
}

pub trait Geometry {
    fn length(&self) -> f64;
}

impl Geometry for Point {
    fn length(&self) -> f64 {
        ((self.x * self.x + self.y * self.y) as f64).sqrt()
    }
}

impl Geometry for Line {
    fn length(&self) -> f64 {
        let dx = self.p2.x - self.p1.x;
        let dy = self.p2.y - self.p1.y;

        ((dx * dx + dy * dy) as f64).sqrt()
    }
}

impl Poly {
    /// Sorts the points of the contour counter clockwise
    pub fn order(&self) -> Option<opencv::core::Vector<Point>> {
        let center = {
            let l = self.0.len() as i32;
            let points_sum = self.0.iter().reduce(|p1, p2| Point {
                x: p1.x + p2.x,
                y: p1.y + p2.y,
            })?;
            Point {
                x: points_sum.x / l,
                y: points_sum.y / l,
            }
        };

        let points = self
            .0
            .iter()
            .map(|p| Point {
                x: p.x - center.x,
                y: p.y - center.y,
            })
            .filter(|p| p.length() > 0.0);

        let angles = points.clone().map(|p| ((p.y as f64) / p.length()).acos());

        let mut pts_with_angles: Vec<_> = self.0.to_vec().iter().cloned().zip(angles).collect();
        pts_with_angles.sort_by(|(_, a), (_, b)| a.total_cmp(b));

        let points: Vec<_> = pts_with_angles.iter().cloned().map(|p| p.0).collect();

        Some(points.into())
    }
}

impl std::cmp::PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        let self_area = opencv::imgproc::contour_area_def(&self.0).unwrap_or(f64::NAN);
        let other_area = opencv::imgproc::contour_area_def(&other.0).unwrap_or(f64::NAN);

        self_area == other_area
    }
}

impl std::cmp::Eq for Poly {}

impl std::cmp::PartialOrd for Poly {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let self_area = opencv::imgproc::contour_area_def(&self.0).ok()?;
        let other_area = opencv::imgproc::contour_area_def(&other.0).ok()?;

        f64::partial_cmp(&self_area, &other_area)
    }
}

impl std::cmp::Ord for Poly {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_area = opencv::imgproc::contour_area_def(&self.0).unwrap_or(0.0);
        let other_area = opencv::imgproc::contour_area_def(&other.0).unwrap_or(0.0);

        f64::total_cmp(&self_area, &other_area)
    }
}

struct Table {
    image: Mat,
    rows: Vec<Vec<opencv::core::Rect>>,
}

impl TryFrom<Mat> for Table {
    type Error = opencv::Error;
    fn try_from(image: Mat) -> opencv::Result<Self> {
        let mut binary = Mat::default();
        opencv::imgproc::threshold(&image, &mut binary, 128.0, 255.0, opencv::imgproc::THRESH_OTSU)?;        
        
        let mut inv = Mat::default();
        opencv::core::bitwise_not_def(&binary, &mut inv)?;
        let binary = inv;

        let horizontal_kernel = opencv::imgproc::get_structuring_element_def(
            opencv::imgproc::MORPH_RECT,
            opencv::core::Size::new(image.size()?.width / 50, 1),
        )?;

        let vertical_kernel = opencv::imgproc::get_structuring_element_def(
            opencv::imgproc::MORPH_RECT,
            opencv::core::Size::new(1, image.size()?.height / 35),
        )?;

        let vh_kernel = opencv::imgproc::get_structuring_element_def(
            opencv::imgproc::MORPH_CROSS,
            opencv::core::Size::new(3, 3),
        )?;
        
        let iterations = 8;

        let mut binary_horizontal = Mat::default();
        opencv::imgproc::morphology_ex(
            &binary,
            &mut binary_horizontal,
            opencv::imgproc::MORPH_OPEN,
            &horizontal_kernel,
            opencv::core::Point::new(-1, -1),
            iterations,
            opencv::core::BORDER_CONSTANT,
            opencv::imgproc::morphology_default_border_value()?,
        )?;

        let mut binary_vertical = Mat::default();
        opencv::imgproc::morphology_ex(
            &binary,
            &mut binary_vertical,
            opencv::imgproc::MORPH_OPEN,
            &vertical_kernel,
            opencv::core::Point::new(-1, -1),
            iterations,
            opencv::core::BORDER_CONSTANT,
            opencv::imgproc::morphology_default_border_value()?,
        )?;

        let mut vh_lines = Mat::default();
        opencv::core::add_weighted_def(
            &binary_vertical,
            0.5,
            &binary_horizontal,
            0.5,
            0.0,
            &mut vh_lines,
        )?;

        let mut not_vh_lines = Mat::default();
        opencv::core::bitwise_not_def(&vh_lines, &mut not_vh_lines)?;

        let mut not_vh_lines_eroded = Mat::default();
        opencv::imgproc::erode(
            &not_vh_lines,
            &mut not_vh_lines_eroded,
            &vh_kernel,
            opencv::core::Point::new(-1, -1),
            2,
            opencv::core::BORDER_CONSTANT,
            opencv::imgproc::morphology_default_border_value()?,
        )?;

        let mut not_vh_lines_threshold = Mat::default();
        opencv::imgproc::threshold(
            &not_vh_lines_eroded,
            &mut not_vh_lines_threshold,
            128.0,
            255.0,
            opencv::imgproc::THRESH_OTSU | opencv::imgproc::THRESH_BINARY,
        )?;

        let mut contours: opencv::core::Vector<opencv::core::Vector<Point>> =
            opencv::core::Vector::new();
        opencv::imgproc::find_contours_def(
            &not_vh_lines_threshold,
            &mut contours,
            opencv::imgproc::RETR_TREE,
            opencv::imgproc::CHAIN_APPROX_SIMPLE,
        )?;

        let boxes: Vec<_> = contours
            .into_iter()
            .map(|c| opencv::imgproc::bounding_rect(&c))
            .filter_map(opencv::Result::ok)
            .sorted_by_key(|b| b.y)
            .collect();

        let mean_height = boxes
            .iter()
            .map(|b| b.height as f64)
            .collect::<average::Mean>()
            .mean();

        let rows: Vec<_> = boxes
            .into_iter()
            .batching(|it| {
                let current = it.clone().next()?;
                let current_max_height = current.y as f64 + mean_height / 2.0;

                // Iterate over boxes, starting with 'current'
                let x = it
                    .take_while(|b| b.y as f64 <= current_max_height)
                    .sorted_by_key(|b| b.x)
                    .collect::<Vec<_>>();

                Some(x)
            })
            .collect();

        Ok(Table { image, rows })
    }
}

pub fn imgproc_pipeline(image: Mat) -> opencv::Result<Mat> {
    let prepared = {
        let mut original_image = Mat::default();
        image.copy_to(&mut original_image)?;

        let mut prepared = Mat::default();
        opencv::imgproc::cvt_color_def(&image, &mut prepared, opencv::imgproc::COLOR_BGR2GRAY)?;

        let image = prepared;
        let mut prepared = Mat::default();
        opencv::imgproc::gaussian_blur(
            &image,
            &mut prepared,
            Size::new(3, 3),
            2.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
        )?;

        let image = prepared;
        let mut prepared = Mat::default();
        opencv::imgproc::threshold(&image, &mut prepared, 128.0, 255.0, opencv::imgproc::THRESH_OTSU)?;        

        // TODO: Why does adaptive threshold perform so poorly?
        // opencv::imgproc::adaptive_threshold(
        //     &image,
        //     &mut prepared,
        //     255.0,
        //     opencv::imgproc::ADAPTIVE_THRESH_GAUSSIAN_C,
        //     opencv::imgproc::THRESH_BINARY,
        //     11,
        //     2.0,
        // )?;

        let image = prepared;
        let mut prepared = Mat::default();
        opencv::photo::fast_nl_means_denoising(&image, &mut prepared, 11.0, 31, 9)?;

        let mut edges = Mat::default();
        opencv::imgproc::canny(&prepared, &mut edges, 50.0, 150.0, 3, false)?;

        let mut contours: opencv::core::Vector<opencv::core::Vector<Point>> =
            opencv::core::Vector::new();
        opencv::imgproc::find_contours_def(
            &edges,
            &mut contours,
            opencv::imgproc::RETR_EXTERNAL,
            opencv::imgproc::CHAIN_APPROX_SIMPLE,
        )?;

        let biggest_rect: opencv::core::Vector<_> = contours
            .iter()
            .map(|c| -> opencv::Result<_> {
                let mut hull: opencv::core::Vector<Point> = opencv::core::Vector::new();
                opencv::imgproc::convex_hull_def(&c, &mut hull)?;

                let mut poly = opencv::core::Vector::new();
                opencv::imgproc::approx_poly_dp(
                    &hull,
                    &mut poly,
                    0.001 * opencv::imgproc::arc_length(&hull, true)?,
                    true,
                )?;

                Ok(poly)
            })
            .filter_map(Result::ok)
            .filter(|c| c.len() == 4)
            .map(|c| Poly(c))
            .max()
            .ok_or(opencv::Error::new(1, "Could not get the biggest contour"))?
            .order()
            .ok_or(opencv::Error {
                code: 1,
                message: "Couldn't do thing".to_owned(),
            })?;

        // FIXME: Why is the order messed up here, it shouldn't be, but that's how it currently works
        let biggest_rect = Rect {
            top_left: biggest_rect.get(2)?,
            top_right: biggest_rect.get(3)?,
            bottom_right: biggest_rect.get(0)?,
            bottom_left: biggest_rect.get(1)?,
        };

        let max_height =
            f64::max(biggest_rect.left().length(), biggest_rect.right().length()) as f32;

        let max_width =
            f64::max(biggest_rect.top().length(), biggest_rect.bottom().length()) as f32;

        let dst = opencv::core::Vector::from_slice(&[
            Point2f::new(0.0, 0.0),
            Point2f::new(max_width - 1.0, 0.0),
            Point2f::new(max_width - 1.0, max_height - 1.0),
            Point2f::new(0.0, max_height - 1.0),
        ]);

        let transform = opencv::imgproc::get_perspective_transform(
            &Into::<opencv::core::Vector<Point2f>>::into(biggest_rect),
            &dst,
            opencv::core::DECOMP_LU,
        )?;

        let image = prepared;
        let mut prepared = Mat::default();
        opencv::imgproc::warp_perspective_def(
            &image,
            &mut prepared,
            &transform,
            opencv::core::Size::new(max_width as i32, max_height as i32),
        )?;
        prepared
    };
    
    let binary = {
        let mut binary = Mat::default();
        opencv::photo::fast_nl_means_denoising(&prepared, &mut binary, 10.0, 7, 21)?;
        
        let image = binary;        
        let mut binary = Mat::default();
        opencv::imgproc::threshold(&image, &mut binary, 128.0, 255.0, opencv::imgproc::THRESH_OTSU)?;        
        
        let mut sharpening_kernel = Mat::from_slice_2d(&[
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ])?;

        let image = binary;
        let mut binary = Mat::default();
        opencv::imgproc::filter_2d_def(&image, &mut binary, -1, &sharpening_kernel)?;

        binary
    };

    let mut ocr = opencv::text::OCRTesseract::create(
        "tessdata/",
        "eng",
        "",
        opencv::text::OEM_DEFAULT,
        opencv::text::PSM_AUTO,
    )?;

    let delta = 1;
    let scaling = 2;
    let table = Arc::new(Table::try_from(binary)?);

    let mut image = Mat::default();
    opencv::imgproc::cvt_color_def(&table.image, &mut image, opencv::imgproc::COLOR_GRAY2BGR)?;

    for row in &table.rows {
        for col in row {
            opencv::imgproc::rectangle_def(&mut image, *col, (255.0, 0.0, 0.0).into())?;
        }
    }

    return Ok(image);

    let first_min_x = table.rows[0][1].x - 10;

    let interesting_rows = table
        .rows
        .iter()
        // FIXME: Bad way of handling unexpected columns
        .filter(|row| row.len() >= 3);

    let names: Vec<_> = interesting_rows
        .clone()
        .map(|row| row[1])
        .filter(|name_col| name_col.width > 10 && name_col.height > 10)
        .map(|name_col| {
            if name_col.x < first_min_x {
                opencv::core::Rect::new(
                    name_col.x - delta,
                    name_col.y - delta,
                    name_col.width - delta,
                    name_col.height - delta,
                )
            } else {
                name_col
            }
        })
        .map(|roi| -> opencv::Result<_> {
            let img = Mat::roi(&table.image, roi)?;
            Ok((img, roi))
        })
        .filter_map(opencv::Result::ok)
        .map(|(img, roi)| -> opencv::Result<_> {
            let new_size = opencv::core::Size::new(roi.width * scaling, roi.height * scaling);

            let mut scaled = Mat::default();
            opencv::imgproc::resize_def(&img, &mut scaled, new_size)?;

            let mut closed = Mat::default();
            let kernel = opencv::imgproc::get_structuring_element_def(
                opencv::imgproc::MORPH_CROSS,
                opencv::core::Size::new(3, 3),
            )?;
            opencv::imgproc::morphology_ex_def(
                &scaled,
                &mut closed,
                opencv::imgproc::MORPH_CLOSE,
                &kernel,
            )?;

            let mut thresholded = Mat::default();
            let _ = opencv::imgproc::threshold(
                &closed,
                &mut thresholded,
                128.0,
                255.0,
                opencv::imgproc::THRESH_BINARY,
            )?;

            let delta = delta * scaling;
            let thresholded = Mat::roi(
                &thresholded,
                opencv::core::Rect::new(
                    delta,
                    delta,
                    thresholded.size()?.width - delta,
                    thresholded.size()?.height - delta,
                ),
            )?;

            Ok(thresholded)
        })
        .filter_map(opencv::Result::ok)
        .filter(|img| !img.empty())
        .map(|mut img| -> opencv::Result<_> {
            let mut name = String::default();
            ocr.run_def(&mut img, &mut name)?;
            Ok(name)
        })
        .filter_map(opencv::Result::ok)
        .collect();

    Ok(image)
}
