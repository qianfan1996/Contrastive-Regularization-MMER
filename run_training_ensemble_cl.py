# -*-coding:utf-8-*-
import torch
import os
import json
import argparse
import numpy as np
from time import localtime, strftime

from models import FeatureEnsemble, Mimax, CPMC, Classify, LoadFeatureEnsemble
from batch_iterator import EnsembleBatchIterator
from data_loader import load_linguistic_dataset, load_spectrogram_dataset, create_batches
from config import LinguisticConfig, AcousticSpectrogramConfig, EnsembleConfig
from utils import log_success, log_major, get_datetime, set_default_tensor, get_device, create_new_diff_emotion_label, create_new_emotion_linguistic_data
from confusion_matrix import ConfusionMatrix

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

MODEL_PATH = "saved_models/INFONCE"
learning_rate = 0.001
num_epochs = 1000
patience = 30
# alpha = 0.5
loss_function = 'INFONCE'
# model_name = 'ensemble_' + loss_function + '_alpha' + str(alpha)
# model_weight_name = 'ensemble_' + loss_function + '_alpha' + str(alpha) + '_model.torch'

alpha_list = [i/10 for i in range(1, 10)]
for alpha in alpha_list:
	model_name = 'ensemble_' + loss_function + '_alpha' + str(alpha) + '_pretrain'
	model_weight_name = 'ensemble_' + loss_function + '_alpha' + str(alpha) + '_pretrain' + '_model.torch'
	tmp_run_path = "tmp/" + model_name + "_" + get_datetime()
	model_weights_path = "{}/{}".format(tmp_run_path, model_weight_name)
	result_path = "{}/result.txt".format(tmp_run_path)

	os.makedirs(tmp_run_path, exist_ok=True)

	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--ensemble_model", type=str, required=True)
	args = parser.parse_args()

	set_default_tensor()

	assert os.path.isfile(args.ensemble_model), "ensemble_model weights file does not exist"
	assert os.path.isfile(args.ensemble_model.replace(".torch", ".json")), "ensemble_model config file does not exist"

	ensemble_cfg_json = json.load(open(args.ensemble_model.replace(".torch", ".json"), "r"))
	acoustic_config, linguistic_config = ensemble_cfg_json['acoustic_config'], ensemble_cfg_json['linguistic_config']
	acoustic_config = AcousticSpectrogramConfig(**acoustic_config)
	linguistic_config = LinguisticConfig(**linguistic_config)
	ensemble_cfg = EnsembleConfig(acoustic_config, linguistic_config)
	ensemble_model = FeatureEnsemble(ensemble_cfg)
	ensemble_model.load_state_dict(torch.load(args.ensemble_model))

	model_load_ensemble = LoadFeatureEnsemble(ensemble_model)
	model_mimax = Mimax()
	model_classify = Classify()

	"""Defining loss and optimizer"""
	optimizer = torch.optim.Adam([{'params':model_classify.parameters()}, {'params':model_mimax.parameters()}, {'params':model_load_ensemble.parameters()}], lr=learning_rate)
	criterion = torch.nn.CrossEntropyLoss()
	criterion = criterion.to(get_device())

	best_test_loss = 999
	best_test_acc = 0
	best_val_loss = 999
	epochs_without_improvement = 0
	"""
	test_iterator_ac, train_iterator_ac = create_batches_no_valid(*load_spectrogram_dataset(), 128)
	test_iterator_li, train_iterator_li = create_batches_no_valid(*load_linguistic_dataset(), 128)
	test_iterator = EnsembleBatchIterator(test_iterator_ac, test_iterator_li)
	train_iterator = EnsembleBatchIterator(train_iterator_ac, train_iterator_li)
	"""
	test_iterator_ac, train_iterator_ac, validation_iterator_ac = create_batches_with_valid(*load_spectrogram_dataset(), 128)
	test_iterator_li, train_iterator_li, validation_iterator_li = create_batches_with_valid(*load_linguistic_dataset(), 128)
	test_iterator = EnsembleBatchIterator(test_iterator_ac, test_iterator_li)
	train_iterator = EnsembleBatchIterator(train_iterator_ac, train_iterator_li)
	validation_iterator = EnsembleBatchIterator(validation_iterator_ac, validation_iterator_li)

	for epoch in range(num_epochs):
		train_iterator.shuffle()
		if epochs_without_improvement == patience:
			break

		# train
		model_load_ensemble.train()
		model_classify.train()
		model_mimax.train()

		train_epoch_loss = 0
		train_conf_mat = ConfusionMatrix(np.zeros((4, 4)))

		for batch, labels in train_iterator():
			optimizer.zero_grad()
			"""
			label = labels.cpu().numpy()
			new_diff_emotion_labels = create_new_diff_emotion_label(label)
			new_diff_emotion_data = create_new_emotion_linguistic_data(new_diff_emotion_labels)
			new_diff_emotion_linguistic = torch.from_numpy(new_diff_emotion_data).float().cuda()
			"""
			batch_ = batch[1].cpu().numpy()
			np.random.shuffle(batch_)
			li_shuffle = torch.from_numpy(batch_).float().cuda()

			# _, linguistic_feature_ = model_encoder((batch[0], new_diff_emotion_linguistic))
			_, linguistic_feature_ = model_load_ensemble((batch[0], li_shuffle))
			acoustic_feature, linguistic_feature = model_load_ensemble(batch)

			predictions = model_classify(acoustic_feature, linguistic_feature).squeeze(1)
			loss1 = criterion(predictions, labels)

			pred_xy = model_mimax(acoustic_feature, linguistic_feature)
			pred_x_y = model_mimax(acoustic_feature, linguistic_feature_)

			if loss_function == 'MINE':
				ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
			elif loss_function == 'BCE':
				ret = torch.mean(torch.log(pred_xy)) + torch.mean(torch.log(1-pred_x_y))
			else:
				ret = criterion(model_cpmc((acoustic_feature[i], linguistic_feature)), torch.tensor([i]))
			loss2 = -ret

			loss = (1-alpha)*loss1 + alpha*loss2
			loss.backward()

			optimizer.step()

			train_epoch_loss += loss1.item()

			train_conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

		train_average_loss = train_epoch_loss / len(train_iterator)
		train_acc = train_conf_mat.accuracy

		# test
		model_load_ensemble.eval()
		model_classify.eval()
		model_mimax.eval()

		val_epoch_loss = 0
		val_conf_mat = ConfusionMatrix(np.zeros((4, 4)))

		with torch.no_grad():
			for batch, labels in validation_iterator():
				acoustic_feature, linguistic_feature = model_load_ensemble(batch)
				predictions = model_classify(acoustic_feature, linguistic_feature).squeeze(1)

				loss = criterion(predictions.float(), labels)
				val_epoch_loss += loss.item()
				val_conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

		val_average_loss = val_epoch_loss / len(test_iterator)
		val_acc = val_conf_mat.accuracy

		if val_average_loss < best_val_loss:
			state = {'model_load_ensemble': model_load_ensemble.state_dict(), 'model_classify': model_classify.state_dict(), 'model_mimax': model_mimax.state_dict()}
			torch.save(state, model_weights_path)
			best_val_loss = val_average_loss
			best_val_acc = val_conf_mat.accuracy
			best_val_unweighted_acc = val_conf_mat.unweighted_accuracy
			epochs_without_improvement = 0
			log_success(" Epoch: {} | Validation loss improved to {:.4f} | Validation accuracy: {:.4f} | Weighted validation accuracy: {:.4f} | train loss: {:.4f} | train accuracy: {:.4f} | saved model to {}.".format(
				epoch, best_val_loss, best_val_acc, best_val_unweighted_acc, train_average_loss, train_acc, model_weights_path))

		epochs_without_improvement += 1

	checkpoint = torch.load(model_weights_path)
	model_load_ensemble.load_state_dict(checkpoint['model_load_ensemble'])
	model_classify.load_state_dict(checkpoint['model_classify'])
	model_mimax.load_state_dict(checkpoint['model_mimax'])

	model_load_ensemble.eval()
	model_classify.eval()
	model_mimax.eval()

	test_epoch_loss = 0
	test_conf_mat = ConfusionMatrix(np.zeros((4, 4)))

	with torch.no_grad():
		for batch, labels in test_iterator():
			acoustic_feature, linguistic_feature = model_load_ensemble(batch)
			predictions = model_classify(acoustic_feature, linguistic_feature).squeeze(1)

			loss = criterion(predictions.float(), labels)
			test_epoch_loss += loss.item()
			test_conf_mat += ConfusionMatrix.from_predictions(predictions, labels)

	test_average_loss = test_epoch_loss / len(test_iterator)
	test_acc = test_conf_mat.accuracy

	result = f'| Run Epoch: {epoch + 1} | Test Loss: {test_average_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Weighted Test Acc: {test_conf_mat.unweighted_accuracy * 100:.2f}%\n Confusion matrix:\n {test_conf_mat}'
	log_major(result)
	with open(result_path, "w") as file:
		file.write(result)

	output_path = "{}/{}_{:.4f}Acc_{:.4f}UAcc_{}".format(MODEL_PATH, model_name, test_acc, test_conf_mat.unweighted_accuracy, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
	os.renames(tmp_run_path, output_path)

