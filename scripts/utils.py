import joblib
import os
import json


    
def samples_to_database():
    self.database.remove()
    self.database.create()
    filenames = os.listdir(self.sample_dir)
    for file_number, filename in enumerate(filenames):
        if self.mode in filename:
            print(f"Processing file number {file_number}/{len(filenames)}")
            with open(os.path.join(self.sample_dir, filename), 'r') as f:
                content = json.loads(f.read())
            X, y = self._accumulate_rows(content)
            self.database.insert(X, y)
            print(f"Processed file {filename}")


def samples_to_dataset(self, output_filename):
    filenames = os.listdir(self.sample_dir)
    data = {
        "features": [],
        "targets": []
    }
    for file_number, filename in enumerate(filenames):
        if self.mode in filename:
            print(f"Processing file number {file_number}/{len(filenames)}")
            with open(os.path.join(self.sample_dir, filename), 'r') as f:
                content = json.loads(f.read())
            X, y = self._accumulate_rows(content)
            data["features"].extend(X)
            data["targets"].extend(y)
            print(f"Processed file {filename}")

    assert data["features"] != []
    assert data["targets"] != []

    if self.operate_on_json:
        json_formatted = json.dumps(data, indent=4)
        with open(os.path.join(self.dataset_dir, output_filename), 'w') as f:
            f.write(json_formatted)
            print(f"Saved dataset {output_filename}")
    else:
        joblib.dump(data, os.path.join(self.dataset_dir, output_filename))               

