#[macro_use]
extern crate criterion;
extern crate ndarray;
use ndarray::Array1;
use criterion::Criterion;
use lara::clustering::cluster::Cluster;
use lara::supervised::supervised::Supervised;
use lara::supervised::knn::KNearestNeighbors;
use lara::clustering::kmeans::{InitAlgorithm, KMeans};
use ndarray::{Array2, Array};
use csv::Reader;


fn load_iris_dataset() -> Array2<f32> {
    let mut rdr = Reader::from_path("benches/iris.csv").unwrap();
    let mut count = 0;
    let mut all_data: Vec<f32> = vec![];
    rdr.records().for_each(|row| {
        let row = row.unwrap();
        count += 1;
        all_data.push(row[0].parse::<f32>().unwrap());
        all_data.push(row[1].parse::<f32>().unwrap());
        all_data.push(row[2].parse::<f32>().unwrap());
    });
    Array::from_shape_vec((149, 3), all_data).unwrap()
}

fn load_iris_dataset_ys() -> Array1<usize> {
    let mut rdr = Reader::from_path("benches/iris_y.csv").unwrap();
        let mut count = 0;
    let mut all_data: Vec<usize> = vec![];
    rdr.records().for_each(|row| {
        let row = row.unwrap();
        count += 1;
        all_data.push(row[0].parse::<f32>().unwrap() as usize);
    });
    Array::from_vec(all_data)
}

fn criterion_benchmark(c: &mut Criterion) {
    let xs2 = load_iris_dataset();
    let xs3 =load_iris_dataset();
    let xs4 = load_iris_dataset();
    let ys4 = load_iris_dataset_ys();
    let xs5 = load_iris_dataset();
    let ys5 = load_iris_dataset_ys();
    c.bench_function("kmeans fit", move |b| {
        b.iter(|| {
            for _ in 0..1 {
                let mut clf = KMeans::new(&xs2, 100, 2, InitAlgorithm::Random);
                clf.fit(&xs2);
            }
        })
    });
    c.bench_function("kmeans predict", move |b| {
        let mut clf = KMeans::new(&xs3, 100, 2, InitAlgorithm::Random);
        clf.fit(&xs3);
        b.iter(|| {
            clf.predict(&xs3);
        })
    });
    c.bench_function("knn fit", move |b| {
        let mut clf = KNearestNeighbors::new(3);
        b.iter(|| {
            clf.fit(&xs4, &ys4);
        })
    });
    c.bench_function("knn predict", move |b| {
        let mut clf = KNearestNeighbors::new(3);
        clf.fit(&xs5, &ys5);
        b.iter(|| {
            clf.predict(&xs5);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
