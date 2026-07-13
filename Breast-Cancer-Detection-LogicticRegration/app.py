from fastapi import FastAPI
import pickle

model_file = pickle.load('model/logistic_model.pkl', )