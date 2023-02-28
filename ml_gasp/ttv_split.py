import json
import multiprocessing
import os
import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from mlinsights.plotting import pipeline2dot
from pymatgen.io.vasp import Poscar
from pyquickhelper.loghelper import run_cmd
from quantumml.featurizers import SoapTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

matplotlib.use("Agg")

print("imports done")

class ModelGA():
    def __init__(self, path, elements, name='elements_descriptor_model', n_cores = 1, str_ext = "poscar", target_ext = "energy"):
        self.n_cores = n_cores #multiprocessing.cpu_count()
        self.path = path
        self.str_ext = str_ext
        self.target_ext = target_ext
        self.elements = elements
        self.name = name #"Cd_Te_soap_SVR"
        self.indexs = [os.path.splitext(f)[0] for f in os.listdir(self.path) if self.str_ext in f]

    def process_entry(self, index):
        """parse files"""

        #entry_name, elem_list, path, str_ext, target_ext = args
        target_file = "{}/{}.{}".format(self.path, index, self.target_ext)
        with open(target_file) as f:
            lines = f.read().splitlines()
        local_energies = [float(line) for line in lines]
        # This dataset has per-atom energies but we won't use them
        energy = np.sum(local_energies)

        structure_file = "{}/{}.{}".format(self.path, index, self.str_ext)
        structure = Poscar.from_file(structure_file).structure
        entry_data = [len(structure), energy]
        for elem in self.elements:
            elem_percent = structure.composition.get_atomic_fraction(elem)
            entry_data.append(elem_percent)
        return structure, entry_data

    def parallel_process_entry(self):
        parsed_data = tqdm(multiprocessing.Pool(self.n_cores).imap(self.process_entry, self.indexs), total=len(self.indexs))
        return list(parsed_data)

    def unpack_parallel_process_entry(self,parsed_data):
        # unpack parallel parsing results
        structure_list, table_data = zip(*parsed_data)
        columns = ["Size", "Total Energy"] + self.elements  # pandas header
        structures = {k: v for k, v in zip(self.indexs, structure_list)}
        df = pd.DataFrame(columns=columns, data=table_data, index=self.indexs)
        return df, structures

    def get_RE(self,df):
        ref_E = {}
        for elem in self.elements:
            pure_entries = df[df[elem] == 1.0]
            pure_entry_energies = pure_entries["Total Energy"].values
            pure_entry_energies /= pure_entries["Size"].values
            minimum_energy = np.min(pure_entry_energies)
            ref_E[elem] = minimum_energy
        return ref_E

    def get_final_df(self,df):

        def get_FE(entry_data, elem_list, ref_E):
            elem_fractions = {elem: entry_data[elem] for elem in elem_list}
            reference_contributions = [
                (ref_E[elem] * elem_fractions[elem]) for elem in elem_list
            ]
            FE = entry_data["Total Energy"] / entry_data["Size"] - np.sum(
                reference_contributions
            )
            return FE
        df["Formation Energy"] = df.apply(get_FE, axis=1, args=(self.elements, self.get_RE(df)))
        return df

    @staticmethod
    def test_train_split(structures, df,training_fraction = 0.8):
        ids = set([name.split('_')[0] for name in structures.keys()])
        entry_ids = sorted(ids)
        shuffled_ids = list(entry_ids)
        np.random.shuffle(shuffled_ids)
        #entry_names = sorted(structures.keys())
        #shuffled_names = list(entry_names)
        #np.random.shuffle(shuffled_names)

        n_train = int(len(shuffled_ids) * training_fraction)

        #training_names = shuffled_names[:n_train]
        #testing_names = shuffled_names[n_train:]
        training_ids = shuffled_ids[:n_train]
        testing_ids = shuffled_ids[n_train:]

        training_inputs = []
        training_outputs = []
        testing_inputs = []
        testing_outputs = []

        for name in structures.keys():
            ind = name.split('_')[0]
            if ind in training_ids:
                training_inputs.append(structures[name])
                training_outputs.append(df.loc[name]["Formation Energy"])
            elif ind in testing_ids:
                #print(f'ind {ind} in test')
                testing_inputs.append(structures[name])
                testing_outputs.append(df.loc[name]["Formation Energy"])
            else:
                print(f'ERROR: {ind} not in ids')
        print(f'len(train_in) = {len(training_inputs)}')
        print(f'len(train_out) = {len(training_outputs)}')
        print(f'len(test_in) = {len(testing_inputs)}')
        print(f'len(test_out) = {len(testing_outputs)}')

        #training_inputs = [structures[name] for name in training_names]
        #training_outputs = df.loc[training_names]["Formation Energy"]

        #testing_inputs = [structures[name] for name in testing_names]
        #testing_outputs = df.loc[testing_names]["Formation Energy"]
        return training_inputs, training_outputs, testing_inputs,testing_outputs

    def test_train_to_json(self,training_inputs, training_outputs, testing_inputs,testing_outputs):
        alni = {}
        ind = 0
        train = {}
        test = {}
        for struct, E in zip(training_inputs, training_outputs):
            temp_s = struct.as_dict()
            temp_d = {"structure": temp_s, "FE": E}
            train[ind] = temp_d
            ind += 1
        alni["training"] = train
        ind = 0
        for struct, E in zip(testing_inputs, testing_outputs):
            temp_s = struct.as_dict()
            temp_d = {"structure": temp_s, "FE": E}
            test[ind] = temp_d
            ind += 1
        alni["testing"] = test
        with open(f"{self.name}.json", "w") as fp:
            json.dump(alni, fp)

    def make_cv(self,cScale=5, gScale=0.001,n_iter=50):

        param_dist = {
            "C": scipy.stats.expon(scale=cScale),
            "gamma": scipy.stats.expon(scale=gScale),
            "kernel": ["rbf"],
        }
        search = RandomizedSearchCV(
            SVR(),
            param_distributions=param_dist,
            cv=5,
            scoring="neg_mean_squared_error",
            n_iter=n_iter,
            n_jobs=self.n_cores,
            verbose=100,
        )
        return search

    def make_pipes(self, desriptor_trans):
        structure_transformer = Pipeline(steps=[("descriptor", desriptor_trans), ("scaler", StandardScaler())])

        model = Pipeline(steps=[("scaler", StandardScaler()), ("model", SVR(verbose=True))])

        search = self.make_cv()

        cv_fit = Pipeline(
            steps=[
                ("structure_transformer", structure_transformer),
                ("Cross_validation", search),
            ]
        )

        full_model = Pipeline(
            steps=[
                ("structure_transformer", structure_transformer),
                ("model", SVR(verbose=True)),
            ]
        )
        return model, cv_fit, full_model, desriptor_trans

    def plot_full_pipe(self, full_model):
        d = {}
        xl = [f"X_{i}" for i in range(4)]
        xl.append("...")
        xl.append("X_n")
        for i in range(4):
            d[f"Structure{i}"] = xl
        d[".  .  ."] = xl
        d["Structure_n"] = xl
        df = pandas.DataFrame(data=d)


        dot = pipeline2dot(full_model, df)
        dot_file = f"{self.name}.dot"
        with open(dot_file, "w", encoding="utf-8") as f:
            f.write(dot)
        cmd = "dot -G=300 -Tpng {0} -o{0}.png".format(dot_file)
        run_cmd(cmd, wait=True, fLOG=print)

    def to_onnx(self,model,d_len):
        initial_type = [("float_input", FloatTensorType([None, d_len]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open(f"{self.name}.onnx", "wb") as f:
            f.write(onx.SerializeToString())

    def plot_learning_curve(self, model, x_train, training_outputs):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=x_train,
            y=training_outputs,
            cv=5,
            n_jobs=self.n_cores,
            train_sizes=np.linspace(0.1, 1.0, 10),
            verbose=1000,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(
            train_sizes,
            train_mean,
            color="blue",
            marker="o",
            markersize=5,
            label="training accuracy",
        )

        plt.fill_between(
            train_sizes,
            train_mean + train_std,
            train_mean - train_std,
            alpha=0.15,
            color="blue",
        )

        plt.plot(
            train_sizes, test_mean, color="green", linestyle="--", marker="s", markersize=5, label="validation accuracy",
        )

        plt.fill_between(
            train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green"
        )

        plt.grid()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.ylim([0.25, 1.0])
        plt.tight_layout()
        plt.savefig(f"./{self.name}_learning_curve.png", dpi=300)
        plt.close()

    def plot_pred(self, pred, testing_outputs, test = True):
        plt.scatter(pred, testing_outputs, alpha=0.2)
        plt.plot([-1, 1], [-1, 1], "k--")
        plt.axis("equal")
        if test:
            plt.savefig(f"./{self.name}_test_pred.png", dpi=300)
        else:
            plt.savefig(f"./{self.name}_train_pred.png", dpi=300)
        plt.close()

    def write_submit(self, train_rmse, test_rmse,featurizer_params, doi, target_property):
        onnx_path = f'./{self.name}.onnx'
        training_data_url = f'http://127.0.0.1:8000/media/{self.name}.json'
        pipeline = f'./{self.name}.dot.png'
        pipeline_pkl_path = f'./{self.name}.pkl'
        learning_curve_path = f'{self.name}_learning_curve.png'
        with open(f'{self.name}.py', 'w+') as fi:
            fi.write('from machine_learning_model.models import *\n')
            fi.write('mlmm = MachineLearningModelManager()\n')
            fi.write(f'model_name = "{self.name.split("_")[-1]}"\n')
            fi.write(f'descriptor_name = "{self.name.split("_")[-2]}"\n')
            fi.write(f'elements = {self.elements}\n')
            fi.write(f'name = "{self.name}"\n')
            fi.write(f'onnx_path = "{onnx_path}"\n')
            fi.write(f'target_property = "{target_property}"\n')
            fi.write(f'train_error = {train_rmse}\n')
            fi.write(f'test_error = {test_rmse}\n')
            fi.write(f'training_data_url = "{training_data_url}"\n')
            fi.write(f'featurizer_params = {featurizer_params}\n')
            fi.write(f'doi = "{doi}"\n')
            fi.write(f'pipeline = "{pipeline}"\n')
            fi.write(f'pipeline_pkl_path = "{pipeline_pkl_path}"\n')
            fi.write(f'learning_curve_path = "{learning_curve_path}"\n')
            fi.write(f'mlmm.create_model(elements, onnx_path, train_error, test_error, featurizer_params, doi, target_property, training_data_url,descriptor_name, model_name,pipeline, pipeline_pkl_path,learning_curve_path)\n')


def main():
    path = "."
    model_name = 'SVR'
    descriptor_name = 'SOAP'
    elements = ['Au', 'Be']
    names = elements.copy()
    names.extend([descriptor_name, model_name])
    name = '_'.join(names)
    n_cores = multiprocessing.cpu_count()
    print(f'n_cores = {n_cores}')
    n_cores = 50
    print('Initialize')
    mga = ModelGA(path, elements,name,n_cores)
    #print(mga.indexs)
    
    print('parsing')
    parsed_data = mga.parallel_process_entry()

    print('Unpacking')
    df_i, structures = mga.unpack_parallel_process_entry(parsed_data)
    df = mga.get_final_df(df_i)

    print('Test/Train split')
    training_inputs, training_outputs, testing_inputs, testing_outputs = mga.test_train_split(structures, df)
    
    featurizer_params = {'species': elements, 'rcut': 7.0, 'nmax': 6, 'lmax': 8}
    soap_trans = SoapTransformer(species=featurizer_params['species'], rcut=featurizer_params['rcut'], nmax=featurizer_params['nmax'], lmax=featurizer_params['lmax'])

    model, , full_modcv_fitel, soap_trans = mga.make_pipes(soap_trans)

    print('Transform')
    x_train = soap_trans.fit_transform(training_inputs)
    x_test = soap_trans.fit_transform(testing_inputs)

    print('Fitting')
    cv_fitted = cv_fit.fit(training_inputs, training_outputs)

    pkl.dump(cv_fitted, open(f"{name}.pkl", "wb"))

    print(cv_fitted.steps[-1][-1].best_params_)

    best = cv_fitted.steps[-1][-1].best_params_
    best_params_ = {"model__C": best["C"],"model__gamma": best["gamma"],"model__kernel": best["kernel"]}

    print('fit model')
    model.set_params(**best_params_)
    model.fit(x_train, training_outputs)

    print('predicting')
    pred_test = model.predict(x_test)
    pred_train = model.predict(x_train)

    test_rmse = np.sqrt(np.mean(np.subtract(pred_test, testing_outputs) ** 2))*1000
    train_rmse = np.sqrt(np.mean(np.subtract(pred_train, training_outputs) ** 2))*1000
    print(f"Train_RMSE = {train_rmse}, Test_RMSE = {test_rmse}")

    mga.plot_full_pipe(full_model)
    mga.to_onnx(model, x_train.shape[-1])
    mga.test_train_to_json(training_inputs, training_outputs, testing_inputs,testing_outputs)
    mga.plot_learning_curve(model, x_train, training_outputs)
    mga.plot_pred(pred_test, testing_outputs, test=True)
    mga.plot_pred(pred_train, training_outputs, test=False)

    target_property = 'FormationEnergy'
    doi = '2.2.2'

    mga.write_submit(train_rmse, test_rmse, featurizer_params, doi, target_property)
    print('DONE')
 
if __name__ == '__main__':
    main()

