from fastapi import FastAPI, Request, Form, Body, BackgroundTasks, Depends, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional

from Kaif.DataSet import RawDataSet, aifData
from Kaif.Metric import DataMetric, ClassificationMetric
from Kaif.Algorithms.Preprocessing import Disparate_Impact_Remover, Learning_Fair_Representation, RW
from Kaif.Algorithms.Inprocessing import Gerry_Fair_Classifier, Meta_Fair_Classifier, Prejudice_Remover
from Kaif.Algorithms.Postprocessing import Calibrated_EqOdds, EqualizedOdds, RejectOption
from Kaif.Algorithms.sota import FairBatch, FairFeatureDistillation, FairnessVAE, KernelDensityEstimator, LearningFromFairness

from sklearn import svm
import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch import optim

from sample import AdultDataset, GermanDataset, CompasDataset, PubFigDataset


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
metrics = None
miti_result = None


@app.get("/", response_class=RedirectResponse)
async def main(request: Request):
    return '/data'


# Data selection
@app.get("/data")
async def data_selection(request: Request):
    #global metrics
    #global miti_result
    #metrics = Metrics()
    #miti_result = Mitigation()

    context = {
        'request': request
    }
    return templates.TemplateResponse('data_selection.html', context)


@app.post("/file", response_class=RedirectResponse)
async def upload_file(file: UploadFile):
    df = pd.read_csv(file.file)
    df.to_csv("custom.csv", index=False)
    return "/original"


class Metrics:
    def __init__(self):
        self.result = None


    def get_metrics(self, dataset):
        print("train model start")
        # 2. Get classification metrics
        privilege = {key: value[0] for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)}
        unprivilege = {key: value[0] for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)}

        ## train model
        model = svm.SVC(random_state=777)
        model.fit(dataset.features, dataset.labels.ravel())

        ## predict
        pred = model.predict(dataset.features)

        ## metric
        metric = ClassificationMetric(dataset=dataset, 
            privilege=privilege, unprivilege=unprivilege, 
            prediction_vector=pred, target_label_name=dataset.label_names[0])
        perfm = metric.performance_measures()

        print("train model end")

        # 3. Make result json
        context = {
            "recall": perfm['TPR'],
            "true_negative_rate": perfm['TNR'],
            "false_positive_rate": perfm['FPR'],
            "false_negative_rate": perfm['FNR'],
            "precision": perfm['PPV'],
            "negative_predictive_value": perfm['NPV'],
            "false_discovery_rate": perfm['FDR'],
            "false_omission_rate": perfm['FOR'],
            "accuracy": perfm['ACC'],
            "error_rate": metric.error_rate(),
            "average_odds_difference": metric.average_odds_difference(),
            "average_abs_odds_difference": metric.average_abs_odds_difference(),
            "selection_rate": metric.selection_rate(),
            "disparate_impact": metric.disparate_impact(),
            "statistical_parity_difference": metric.statistical_parity_difference(),
            "generalized_entropy_index": metric.generalized_entropy_index(),
            "theil_index": metric.theil_index(),
            "equal_opportunity_difference": metric.equal_opportunity_difference()
        }

        self.result = context


    def get_state(self):
        return self.result


metrics = Metrics()


class Mitigation:
    def __init__(self):
        self.result = None

    def get_metrics(self, dataset, method_id):
        print("Mitigation start")
        if method_id == 1:  # Disparate impact remover
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

            # Split the dataset
            dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

            fair_mod = Disparate_Impact_Remover(rep_level=0.5, sensitive_attribute=dataset_train.protected_attribute_names[0])
            transf_dataset = fair_mod.fit_transform(dataset_train)

            # Train
            model = svm.SVC(random_state=777)
            model.fit(transf_dataset.features, transf_dataset.labels.ravel())

            # Prediction
            pred = model.predict(dataset_test.features)

        elif method_id == 2:  # Learning fair representation
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

            # Split the dataset
            dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

            fair_mod = Learning_Fair_Representation(unprivileged_groups=[unprivilege[0]], privileged_groups=[privilege[0]])
            transf_dataset = fair_mod.fit_transform(dataset_train)
            transf_dataset.labels = dataset_train.labels

            # Train
            model = svm.SVC(random_state=777)
            model.fit(transf_dataset.features, transf_dataset.labels.ravel())

            # Prediction
            pred = model.predict(dataset_test.features)

        elif method_id == 3:  # Reweighing
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

            # Split the dataset
            dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

            fair_mod = RW(unprivileged_groups=unprivilege, privileged_groups=privilege)
            transf_dataset = fair_mod.fit_transform(dataset_train)
            transf_dataset.labels = dataset_train.labels

            # Train
            model = svm.SVC(random_state=777)
            model.fit(transf_dataset.features, transf_dataset.labels.ravel())

            # Prediction
            pred = model.predict(dataset_test.features)

        #elif method_id == 4:  # Adversarial debiasing
            #pass
        elif method_id == 5:  # Gerry fair classifier
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

            # Split the dataset
            dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

            gfc = Gerry_Fair_Classifier()
            gfc.fit(dataset_train)

            # Train
            transf_dataset = gfc.predict(dataset_test)

            # Prediction
            pred = transf_dataset.labels

        elif method_id == 6: # Meta fair classifier
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

            # Split the dataset
            dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

            mfc = Meta_Fair_Classifier()
            mfc = mfc.fit(dataset_train)

            # Train
            transf_dataset = mfc.predict(dataset_test)

            # Prediction
            pred = transf_dataset.labels
            
        elif method_id == 7:  # Prejudice remover
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

            # Split the dataset
            dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

            pr = Prejudice_Remover()
            pr.fit(dataset_train)

            # Train
            transf_dataset = pr.predict(dataset_test)

            # Prediction
            pred = transf_dataset.labels
            
        elif method_id == 8:  # Fair batch
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset.protected_attribute_names, dataset.unprivileged_protected_attributes)]

            # Split the dataset
            dataset_train, dataset_test = dataset.split([0.7], shuffle=True)

            protected_label = dataset_train.protected_attribute_names[0]
            protected_idx = dataset_train.feature_names.index(protected_label)
            biased = dataset_train.features[:, protected_idx]

            # RawDataSet
            train_data = RawDataSet(x=dataset_train.features, y=dataset_train.labels, z=biased)

            # Prediction
            batch_size = 256
            alpha = 0.1
            fairness = 'eqodds'
            model, cls2val, _ = FairBatch.train(train_data, batch_size, alpha, fairness)
            pred = FairBatch.evaluation(model, dataset_test, cls2val)

            # Transformed dataset
            transf_dataset = dataset_test.copy(deepcopy=True)
            transf_dataset.labels = np.array(pred).reshape(len(pred), -1)

        elif method_id == 9:
            # Fair feature distillation (Image only)
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names, dataset['aif_dataset'].privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names, dataset['aif_dataset'].unprivileged_protected_attributes)]

            # Flatten the images
            fltn_img = np.array([img.ravel() for img in dataset['image_list']], dtype='int')

            # Split the dataset
            dataset_train, dataset_test = dataset['aif_dataset'].split([0.7], shuffle=True)

            protected_label = dataset_train.protected_attribute_names[0]
            protected_idx = dataset_train.feature_names.index(protected_label)
            biased = dataset_train.features[:, protected_idx]

            # RawDataSet
            rds = RawDataSet(x=fltn_img, y=dataset['target'], z=dataset['bias'])
            #train_data = RawDataSet(x=train_img, y=train_target, z=train_bias)
            #test_data = RawDataSet(x=test_img, y=test_target, z=test_bias)

            # Train
            n_epoch = 20
            batch_size = 64
            learning_rate = 0.01
            image_shape = (3, 64, 64)
            ffd = FairFeatureDistillation.FFD(rds, n_epoch, batch_size, learning_rate, image_shape)
            ffd.train_teacher()
            ffd.train_student()

            # Prediction
            pred = ffd.evaluation()

            # Make aifData for test
            test_X = ffd.test_dataset.X.reshape(len(ffd.test_dataset), -1).cpu().detach().numpy()
            test_y = ffd.test_dataset.y.cpu().detach().numpy()
            test_z = ffd.test_dataset.z.cpu().detach().numpy()
            df = pd.DataFrame(test_X)
            df[protected_label] = test_z
            df[dataset['aif_dataset'].label_names[0]] = test_y

            dataset_test = aifData(df=df, 
                label_name=dataset['aif_dataset'].label_names[0], favorable_classes=[dataset['aif_dataset'].favorable_label],
                protected_attribute_names=dataset['aif_dataset'].protected_attribute_names, privileged_classes=dataset['aif_dataset'].privileged_protected_attributes)


        elif method_id == 10:  # Fair VAE (Image only)
            # Make privileged group and unprivileged group
            privilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names, dataset['aif_dataset'].privileged_protected_attributes)]
            unprivilege = [{key: value[0]} for key, value in zip(dataset['aif_dataset'].protected_attribute_names, dataset['aif_dataset'].unprivileged_protected_attributes)]

            # Flatten the images
            fltn_img = np.array([img.ravel() for img in dataset['image_list']], dtype='int')

            # Split the dataset
            dataset_train, dataset_test = dataset['aif_dataset'].split([0.7], shuffle=True)
            
            protected_label = dataset_train.protected_attribute_names[0]
            protected_idx = dataset_train.feature_names.index(protected_label)
            biased = dataset_train.features[:, protected_idx]

            # RawDataSet
            rds = RawDataSet(x=fltn_img, y=dataset['target'], z=dataset['bias'])
            #train_data = RawDataSet(x=dataset_train.features, y=dataset_train.labels.ravel(), z=biased)
            #test_data = RawDataSet(x=dataset_test.features, y=dataset_test.labels.ravel(), z=biased)

            # Train
            z_dim = 20
            batch_size = 32
            num_epochs = 20
            image_shape=(3, 64, 64)
            fvae = FairnessVAE.FairnessVAE(rds, z_dim, batch_size, num_epochs, image_shape=image_shape)
            fvae.train_upstream()
            fvae.train_downstream()

            # Prediction
            pred = fvae.evaluation()

            # Make aifData for test
            test_X = ffd.test_dataset.X.reshape(len(ffd.test_dataset), -1).cpu().detach().numpy()
            test_y = ffd.test_dataset.y.cpu().detach().numpy()
            test_z = ffd.test_dataset.z.cpu().detach().numpy()
            df = pd.DataFrame(test_X)
            df[protected_label] = test_z
            df[dataset['aif_dataset'].label_names[0]] = test_y

            dataset_test = aifData(df=df, 
                label_name=dataset['aif_dataset'].label_names[0], favorable_classes=[dataset['aif_dataset'].favorable_label],
                protected_attribute_names=dataset['aif_dataset'].protected_attribute_names, privileged_classes=dataset['aif_dataset'].privileged_protected_attributes)


        elif method_id == 11:  # Kernel density_estimation
            
            protected_label = dataset_train.protected_attribute_names[0]
            protected_idx = dataset_train.feature_names.index(protected_label)
            biased = dataset_train.features[:, protected_idx]

            # RawDataSet
            train_data = RawDataSet(x=dataset_train.features, y=dataset_train.labels.ravel(), z=biased)
            test_data = RawDataSet(x=dataset_test.features, y=dataset_test.labels.ravel(), z=biased)

            # Train
            fairness_type = 'DP'
            batch_size = 64
            n_epoch = 20
            learning_rate = 0.01
            kde = KernelDensityEstimator.KDEmodel(train_data, fairness_type, batch_size, n_epoch, learning_rate)
            kde.train()

            # Prediction
            pred = kde.evaluation(test_data)

        elif method_id == 12:  # Learning from fairness (Image only)
            
            protected_label = dataset_train.protected_attribute_names[0]
            protected_idx = dataset_train.feature_names.index(protected_label)
            biased = dataset_train.features[:, protected_idx]

            # RawDataSet
            train_data = RawDataSet(x=dataset_train.features, y=dataset_train.labels.ravel(), z=biased)
            test_data = RawDataSet(x=dataset_test.features, y=dataset_test.labels.ravel(), z=biased)

            # Train
            n_epoch = 20
            batch_size = 64
            learning_rate = 0.01
            image_shape = (3, 64, 64)
            lff = LearningFromFairness.LfFmodel(train_data, n_epoch, batch_size, learning_rate, image_shape)
            lff.train()

            # Prediction
            pred = lff.evaluate(dataset=test_data)

        elif method_id == 13:  # Calibrated equalized odds
            # Train
            model = svm.SVC(random_state=777)
            model.fit(dataset_train.features, dataset_train.labels.ravel())

            # Prediction
            pred = model.predict(dataset_test.features)
            dataset_test_pred = dataset_test.copy(deepcopy=True)
            dataset_test_pred.labels = np.array(pred).reshape(len(pred), -1)

            # Post-processing
            cpp = Calibrated_EqOdds([unprivilege[0]], [privilege[0]])
            cpp = cpp.fit(dataset_test, dataset_test_pred)

            # Re-prediction
            pred_dataset = cpp.predict(dataset_test_pred)
            pred = pred_dataset.scores

        elif method_id == 14:  # Equalized odds
            # Train
            model = svm.SVC(random_state=777)
            model.fit(dataset_train.features, dataset_train.labels.ravel())

            # Prediction
            pred = model.predict(dataset_test.features)
            dataset_test_pred = dataset_test.copy(deepcopy=True)
            dataset_test_pred.labels = np.array(pred).reshape(len(pred), -1)

            # Post-processing
            eqodds = EqualizedOdds([unprivilege[0]], [privilege[0]])
            eqodds = eqodds.fit(dataset_test, dataset_test_pred)

            # Re-prediction
            pred_dataset = eqodds.predict(dataset_test_pred)
            pred = pred_dataset.scores

        elif method_id == 15:  # Reject option
            # Train
            model = svm.SVC(random_state=777)
            model.fit(dataset_train.features, dataset_train.labels.ravel())

            # Prediction
            predict = model.predict(dataset_test.features)
            dataset_test_pred = dataset_test.copy(deepcopy=True)
            dataset_test_pred.labels = np.array(pred).reshape(len(pred), -1)

            # Post-processing
            ro = RejectOption([unprivilege[0]], [privilege[0]])
            ro = ro.fit(dataset_test, dataset_test_pred)

            # Re-prediction
            pred_dataset = ro.predict(dataset_test_pred)
            pred = pred_dataset.scores
            
        else:
            print("ERROR!!")

        ## metric
        transf_metric = ClassificationMetric(dataset=dataset_test, 
            privilege=privilege, unprivilege=unprivilege, 
            prediction_vector=pred, target_label_name=dataset_test.label_names[0])
        perfm = transf_metric.performance_measures()
        print("Mitigation end")

        # 3. Make result
        context = {
            "recall": perfm['TPR'],
            "true_negative_rate": perfm['TNR'],
            "false_positive_rate": perfm['FPR'],
            "false_negative_rate": perfm['FNR'],
            "precision": perfm['PPV'],
            "negative_predictive_value": perfm['NPV'],
            "false_discovery_rate": perfm['FDR'],
            "false_omission_rate": perfm['FOR'],
            "accuracy": perfm['ACC'],
            "error_rate": transf_metric.error_rate(),
            "average_odds_difference": transf_metric.average_odds_difference(),
            "average_abs_odds_difference": transf_metric.average_abs_odds_difference(),
            "selection_rate": transf_metric.selection_rate(),
            "disparate_impact": transf_metric.disparate_impact(),
            "statistical_parity_difference": transf_metric.statistical_parity_difference(),
            "generalized_entropy_index": transf_metric.generalized_entropy_index(),
            "theil_index": transf_metric.theil_index(),
            "equal_opportunity_difference": transf_metric.equal_opportunity_difference()
        }

        self.result = context

    def get_state(self):
        return self.result


miti_result = Mitigation()


# Original Metrics
# Request: form data (Data id)
# Response: Bias metrics (json)
@app.post("/original", response_class=RedirectResponse)
async def original_metrics(request: Request, background_tasks: BackgroundTasks, data_name: Optional[str] = Form(None)):
    global metrics
    global miti_result
    metrics = Metrics()
    miti_result = Mitigation()

    # 1. Get data metrics
    if data_name == 'compas':
        data = CompasDataset()
    elif data_name == 'german':
        data = GermanDataset()
    elif data_name == 'adult':
        data = AdultDataset()
    elif data_name == 'pubfig':
        pubfig = PubFigDataset()
        if not os.path.isdir('./Sample/pubfig'):
            pubfig.download()
            return 'There is no image data on your local. We will download pubfig dataset images from source. Please wait a lot of times. After downloaing the images, you can check images on ./Sample/pubfig directory'
        dataset = pubfig.to_dataset()
        data = dataset['aif_dataset']
    else: # Custom file: data_name = filename
        df = pd.read_csv("custom.csv")
        data = aifData(df=df, label_name='Target', favorable_classes=[1],
            protected_attribute_names=['Bias'], privileged_classes=[[1]])
        #os.remove("custom.csv")
        
    background_tasks.add_task(metrics.get_metrics, data)

    return '/original/{}'.format(data_name)


@app.get("/original/{data_name}")
@app.post("/original/{data_name}")
async def check_metrics(request: Request, data_name: str):
    if not metrics.result:
        return "Processing now...Please wait some minutes..."
    else:
        context = {
            "request": request,
            "data_name": data_name,
            "recall": metrics.result['recall'],
            "true_negative_rate": metrics.result['true_negative_rate'],
            "false_positive_rate": metrics.result['false_positive_rate'],
            "false_negative_rate": metrics.result['false_negative_rate'],
            "precision": metrics.result['precision'],
            "negative_predictive_value": metrics.result['negative_predictive_value'],
            "false_discovery_rate": metrics.result['false_discovery_rate'],
            "false_omission_rate": metrics.result['false_omission_rate'],
            "accuracy": metrics.result['accuracy'],
            "error_rate": metrics.result['error_rate'],
            "average_odds_difference": metrics.result['average_odds_difference'],
            "average_abs_odds_difference": metrics.result['average_abs_odds_difference'],
            "selection_rate": metrics.result['selection_rate'],
            "disparate_impact": metrics.result['disparate_impact'],
            "statistical_parity_difference": metrics.result['statistical_parity_difference'],
            "generalized_entropy_index": metrics.result['generalized_entropy_index'],
            "theil_index": metrics.result['theil_index'],
            "equal_opportunity_difference": metrics.result['equal_opportunity_difference']
        }
        return templates.TemplateResponse('metrics.html', context=context)


# Select a algorithm
@app.get("/algorithm/{data_name}")
async def select_algorithm(request: Request, data_name: str):
    context = {
        "request": request,
        "data_name": data_name
    }
    return templates.TemplateResponse("algorithm_selection.html", context=context)


# Mitigation Result
# Request: form data (Algorithm id, Data id)
# Response: Comparing metrics (json)
@app.post("/mitigation/{data_name}", response_class=RedirectResponse)
async def compare_metrics(request: Request, background_tasks: BackgroundTasks, data_name: str, algorithm: int = Form(...)):
    # 1. Load original metrics (with task_id)

    # 2. Get mitigated result
    if data_name == 'compas':
        data = CompasDataset()
    elif data_name == 'german':
        data = GermanDataset()
    elif data_name == 'adult':
        data = AdultDataset()
    elif data_name == 'pubfig':
        pubfig = PubFigDataset()
        data = pubfig.to_dataset()
    else:  # Custom file: data_name = filename
        df = pd.read_csv("custom.csv")
        data = aifData(df=df, label_name='Target', favorable_classes=[1],
            protected_attribute_names=['Bias'], privileged_classes=[[1]])
        os.remove("custom.csv")

        #print("Error!!! The selected data is not proper.")
        #return "Error!!! The selected data is not proper."

    # 3. Make result json
    background_tasks.add_task(miti_result.get_metrics, dataset=data, method_id=algorithm)

    return f"/mitigation/{data_name}/{algorithm}"


@app.post("/mitigation/{data_name}/{algo_id}")
@app.get("/mitigation/{data_name}/{algo_id}")
async def get_mitigated_result(request: Request, data_name: str, algo_id: int):
    if not miti_result.get_state():
        return "Processing mitigation...Please wait some minutes..."

    context = {
        'request': request,
        'data_name': data_name,
        'algo_id': algo_id,
        'original': metrics.result,
        'mitigated': miti_result.result
    }
    return templates.TemplateResponse('compare.html', context=context)