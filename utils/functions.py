from sklearn.datasets import make_moons, make_circles, make_blobs


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def generate_data(dataset, n_samples, train_noise, test_noise, n_classes):
    if dataset == "spirals":
        x_train, y_train = make_moons(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_moons(n_samples=n_samples, noise=test_noise)
    elif dataset == "circles":
        x_train, y_train = make_circles(n_samples=n_samples, noise=train_noise)
        x_test, y_test = make_circles(n_samples=n_samples, noise=test_noise)
    elif dataset == "blobs":
        x_train, y_train = make_blobs(
            n_features=2,
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=train_noise * 47 + 0.57,
            random_state=123,
        )
        x_test, y_test = make_blobs(
            n_features=2,
            n_samples=n_samples // 2,
            centers=2,
            cluster_std=test_noise * 47 + 0.57,
            random_state=123,
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test
