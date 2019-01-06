#![warn(missing_debug_implementations)]

extern crate multimap;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_parallel;
extern crate rayon;

pub mod clustering {
    pub mod cluster;
    pub mod kmeans;
}

pub mod supervised {
	pub mod supervised;
	pub mod knn;
}

pub mod regression {
    use ndarray::Array2;
    use ndarray_linalg::Inverse;

    /// ```
    /// use lara::regression::least_squares;
    /// use ndarray::arr2;
    /// let xs = arr2(&[[0.23, 1.20, 9.300], [-3.2, 1.8, 2.399]]);
    /// let ys = arr2(&[[1.], [2.]]);
    /// let ground_truth = arr2(&[[0.75], [4.0], [-0.25]]);
    /// let result = least_squares(&xs, &ys);
    /// assert_eq!(ground_truth, result.unwrap());
    pub fn least_squares(
        xs: &Array2<f32>,
        ys: &Array2<f32>,
    ) -> Result<Array2<f32>, ndarray_linalg::error::LinalgError> {
        let xt = xs.t();
        let xtx = xt.dot(xs);
        let xty = xt.dot(ys);
        let xtxi = xtx.inv()?;
        Ok(xtxi.dot(&xty))
    }

}
