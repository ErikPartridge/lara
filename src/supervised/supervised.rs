use ndarray::{Array2, Array1};
pub trait Supervised {
	///
    fn fit(&mut self, xs: &Array2<f32>, ys: &Array1<usize>);
    ///
    fn predict(&self, xs: &Array2<f32>) -> Array1<usize>;

}