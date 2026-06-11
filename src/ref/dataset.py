def generate_images_corr(num, img_size, signal, seed=0):
    rng = np.random.default_rng(seed=seed)
    cov = [[] for _ in range(img_size * img_size)]
    for i in range(img_size * img_size):
        for j in range(img_size * img_size):
            cov[i].append(np.abs(i - j))
    cov = np.array(cov)
    cov = np.power(0.5, cov)

    images = rng.multivariate_normal(np.zeros(img_size * img_size), cov, num)
    images = images.reshape(num, 1, img_size, img_size)
    for i in range(num):
        abnormal_size = img_size // 3
        abnormal_x = rng.integers(0, img_size - abnormal_size)
        abnormal_y = rng.integers(0, img_size - abnormal_size)
        images[
            i,
            0,
            abnormal_x : abnormal_x + abnormal_size,
            abnormal_y : abnormal_y + abnormal_size,
        ] += signal
    return images, cov


def generate_images_non_iid(image_size, distribution, distance, num_samples):
    images = []
    rvs = standardized_rv_at_distance(
        distribution=distribution, distance=distance, metric="wasserstein"
    )
    for _ in range(num_samples):
        images.append(rvs.rvs(size=(1, image_size, image_size)))
    images = np.array(images)
    images = images.reshape(num_samples, 1, image_size, image_size)
    return images
