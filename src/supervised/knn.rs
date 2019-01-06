use crate::supervised::supervised::Supervised;
use ndarray::{arr2, Array2,arr1, Array, Array1};
use std::cmp::Ordering;

pub enum NeighborWeights {
	Uniform
}

#[derive(Debug, Clone, PartialEq)]
pub struct KNearestNeighbors {
	neighbors: usize,
	xs: Array2<f32>,
	ys: Array1<usize>
}

impl KNearestNeighbors {
	pub fn new(neighbors: usize) -> KNearestNeighbors {
		KNearestNeighbors {
			neighbors,
			xs: arr2(&[[]]),
			ys: arr1(&[])

		}
	}

	fn distance(&self, x: &Array1<f32>, point: &Array1<f32>) -> f32 {
        let difference = x - point;
        difference.dot(&difference.t())
    }

}

impl Supervised for KNearestNeighbors {
	fn fit(&mut self, xs: &Array2<f32>, ys: &Array1<usize>) {
		self.xs = xs.to_owned();
		self.ys = ys.to_owned();
	}

	fn predict(&self, inputs: &Array2<f32>) -> Array1<usize> {
		let mut result = vec![];
		for row in inputs.genrows() {
			let mut neighbors : Vec<(f32, usize)> = vec![];
			for (index, neighbor) in self.xs.genrows().into_iter().enumerate() {
				let distance = self.distance(&row.to_owned(), &neighbor.to_owned());
				if neighbors.len() < self.neighbors {
					neighbors.push((distance, self.ys[index]));
				} else if neighbors[neighbors.len() - 1].0 < distance {
					neighbors[self.neighbors - 1] = (distance, self.ys[index]);
					neighbors.sort_unstable_by(|(_ai, an), (_bi, bn)| {
						if an < bn {
							Ordering::Less
						} else if an == bn {
							Ordering::Equal
						} else {
							Ordering::Greater
						}
					});
				}
			}
			let neighbors : Vec<usize> = neighbors.into_iter().map(|(_, y)| {
				y
			}).collect();
			result.push(neighbors[neighbors.len() / 2]);
		}
		Array::from_vec(result)
	}
}