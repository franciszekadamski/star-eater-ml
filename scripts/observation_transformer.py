import numpy as np


class ObservationTransformer:
    def __init__(self, modes=["MIN", "ALL"]):
        self.modes = modes


    def transform(self, input_sample, trim=3):
        samples = []
        for mode in self.modes:
            if mode == "MIN":
                sample = ObservationTransformer.min_transform(input_sample, trim)
                samples.append(sample)

            if mode == "ALL":
                sample = ObservationTransformer.all_transform(input_sample, trim)
                samples.append(sample)

        assert samples != []
        return samples


    @staticmethod
    def transform_single(input_sample, mode, trim):
        input_sample = list(input_sample)
        if mode == "MIN":
            sample = ObservationTransformer.min_transform(input_sample, trim)
        elif mode == "ALL":
            sample = ObservationTransformer.all_transform(input_sample, trim)

        return sample

    # TODO delay
    @staticmethod
    def min_transform(input_sample, trim):
        result = {
            "observations": [],
            "actions": []
        }
        
        for record in input_sample[:-trim]:
            result["actions"].append(record["out"])
            stars = np.asarray(record["in"]["stars"][1:])
            stars = np.abs(stars).min(axis=0)
            rotation = np.asarray(record["in"]["rotation"])
            stars_and_rotation = np.concatenate((stars, rotation)).tolist()
            result["observations"].append(stars_and_rotation)

        return result


    # TODO delay
    @staticmethod
    def all_transform(input_sample, trim):
        result = {
            "observation": [],
            "action": []
        }
        
        for record in input_sample[:-trim]:
            result["actions"].append(record["out"])
            stars = np.asarray(record["in"]["stars"][1:])
            stars = np.abs(stars).flatten()
            rotation = np.asarray(record["in"]["rotation"])
            stars_and_rotation = np.concatenate((stars, rotation)).tolist()
            result["observations"].append(stars_and_rotation)

        return result

