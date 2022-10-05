ABOUT:
Third place solution for "Autonomous Shopper Prediction by Cape AI"
[LINK] https://zindi.africa/competitions/indabax-south-africa-2021

INSTRUCTIONS:
Open the notebook titled "run" and run all the cells from start to finish. To reproduce our submission please see "ensamble.csv". Expected runtime is around 1 hour.

CONTENTS:
-> enviroment.yml
-> graph.py
-> model_att.py
-> model_trans.py
-> tgcn.py
-> run.ipynb
-> SampleSubmission.csv
-> Train.csv
-> Train_Target.csv
-> Test.csv

The following files and directories will be created:
-> ctr-gcn-skip-mini-att.csv
-> ctr-gcn-skip-mini-trans.csv
-> ensamble.csv
-> dataloader_test.pt
-> dataloader_train.pt
-> dataloader_val.pt
-> test_dataset_ex.pkl
-> train_dataset_ex.pkl
-> val_dataset_ex.pkl
-> lightning_logs/

"Lightning_logs" will contain tensorboard logs for model 1 and model 2 (in directories "version_0" and "version_1" respectively) as well as checkpointed models (we use the last model for our preds (last.ckpt), but we also save the model which achieved the highest val AUC).

AUTHORS:
Geoffrey Frost, Kevin Eloff, Matthew Baas
