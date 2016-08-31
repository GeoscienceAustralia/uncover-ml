

def local_learn_model(x_all, targets_all, config):
    model = modelmaps[config.algorithm](**config.algorithm_args)
    y = targets_all.observations

    if mpiops.chunk_index == 0:
        apply_multiple_masked(model.fit, (x_all, y),
                              kwargs={'fields': targets_all.fields})
    model = mpiops.comm.bcast(model, root=0)
    return model


def local_rank_features(image_chunk_sets, transform_sets, targets_all, config):

    # Determine the importance of each feature
    feature_scores = {}
    # get all the images
    all_names = []
    for c in image_chunk_sets:
        all_names.extend(list(c.keys()))
    all_names = sorted(list(set(all_names)))  # make unique

    for name in all_names:
        transform_sets_leaveout = copy.deepcopy(transform_sets)
        final_transform_leaveout = copy.deepcopy(config.final_transform)
        image_chunks_leaveout = [copy.copy(k) for k in image_chunk_sets]
        for c in image_chunks_leaveout:
            if name in c:
                c.pop(name)

        fname = name.rstrip(".tif")
        log.info("Computing {} feature importance of {}"
                 .format(config.algorithm, fname))

        x = pipeline.transform_features(image_chunks_leaveout,
                                        transform_sets_leaveout,
                                        final_transform_leaveout)
        x_all = gather_features(x)

        results = pipeline.local_crossval(x_all, targets_all, config)
        feature_scores[fname] = results

    # Get the different types of score from one of the outputs
    # TODO make this not suck
    measures = list(next(feature_scores.values().__iter__()).scores.keys())
    features = sorted(feature_scores.keys())
    scores = np.empty((len(measures), len(features)))
    for m, measure in enumerate(measures):
        for f, feature in enumerate(features):
            scores[m, f] = feature_scores[feature].scores[measure]
    return measures, features, scores

