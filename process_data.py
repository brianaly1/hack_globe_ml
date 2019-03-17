import csv
import argparse
import numpy as np
import pickle
import os

def shuffle(X,Y):
  X = np.array(X)
  Y = np.array(Y)
  p = np.random.permutation(len(Y))
  return(X[p],Y[p])

def process_csv(file_name):
  with open(file_name, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    symptom_count = 0
    diag_count = 0
    data_symptoms = {}
    data_counts = {}
    symptom_id = {}
    diag_ids = {}
    most_recent = ""
    for row in csv_reader:
      if row["Symptoms"] not in symptom_id:
        symptom_id[row["Symptoms"]] = symptom_count
        symptom_count+=1
      if row["Disease"] != "":
      	most_recent = row["Disease"]
      if most_recent in data_symptoms:
      	data_symptoms[most_recent].append(row["Symptoms"])
      else:
      	data_symptoms[most_recent] = [row["Symptoms"]]
      	data_counts[most_recent] = int(row["Count"])
      	diag_ids[most_recent] = diag_count
      	diag_count += 1
      line_count += 1
    print(f'Processed {line_count} lines.')
    return data_symptoms, data_counts, symptom_id, diag_ids

def get_dataset(data_symptoms, data_counts, symptom_id, diag_id):
	symptom_ids = []
	diagnoses_ids = []
	diseases = []
	num_classes = len(data_symptoms.keys())
	for disease in data_symptoms:
		diseases.append(disease)
		diagnoses_ids.append(diag_id[disease])
		symptom_ids.append([symptom_id[sym] for sym in data_symptoms[disease]])
	inputs = [sum([np.eye(len(symptom_id))[sym_id] for sym_id in sym_ids]) for sym_ids in symptom_ids]
	targets = [np.eye(len(diagnoses_ids))[d_id] for d_id in diagnoses_ids]
	duped_inps = []
	duped_targets = []
	counts = []
	for idx,diagnosis in enumerate(diseases):
	  counts.append(data_counts[diagnosis])
	  duped_inps.extend([inputs[idx]]*data_counts[diagnosis])
	  duped_targets.extend([targets[idx]]*data_counts[diagnosis])
	print("Sum: {}".format(sum(counts)))
	return duped_inps,duped_targets

def distort(inputs):
  new_input = []
  for inp in inputs:
    ran_num = np.random.uniform(0.0,1.0)
    # 75% chance
    if ran_num < 0.95:
      indx_list = []
      for idx,num in enumerate(inp):
        if num==1:
          indx_list.append(idx)
      ran_idx = np.random.randint(0, len(indx_list), dtype='l')
      ran_idx = indx_list[ran_idx]
      new_inp = inp.copy()
      new_inp[ran_idx] = 0
      new_input.append(new_inp)
      continue
    new_inp = inp.copy()
    new_input.append(new_inp)

  return new_input

def save_pickles(distorted_inputs, targets, save_dir, inp_file_name, tar_file_name):
	pickle.dump(distorted_inputs, open(os.path.join(save_dir,inp_file_name), "wb"))
	pickle.dump(targets, open(os.path.join(save_dir,tar_file_name), "wb"))

parser = argparse.ArgumentParser(description='Data Processor')
parser.add_argument('--file_name', required=True)
parser.add_argument('--save_dir', required=True)

args = parser.parse_args()
data_symptoms, data_counts, symptom_id, diag_id = process_csv(args.file_name)
print(len(data_counts))
inputs,targets = get_dataset(data_symptoms, data_counts, symptom_id, diag_id)
print(len(inputs))
print(len(targets))
distorted_inputs = distort(inputs)
distorted_inputs,targets = shuffle(distorted_inputs, targets)
save_pickles(distorted_inputs[:-4096,:], targets[:-4096,:], args.save_dir, "inputs.p", "targets.p")
save_pickles(distorted_inputs[-4096:,:], targets[-4096:,:], args.save_dir, "inputs_val.p", "targets_val.p")
print(len(distorted_inputs[:-4096,:]))
print(len(distorted_inputs[-4096:,:]))


