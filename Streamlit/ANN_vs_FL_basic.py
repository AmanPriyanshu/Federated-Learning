import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

client_epochs = 5
aggregation_epochs = 10

@st.cache
def get_data(n=50, r=0.5, step=4):
	np.random.seed(42)
	x_base = np.array([i*np.pi/180 for i in range(-int(360*r), int(360*r), step)])
	y_base = np.array([i/n for i in range(-n, n, 2)])
	x, y = [], []
	for i in x_base:
		for j in y_base:
			x.append([i, j])
			y.append(0 if np.sin(i)>j else 1)
	x, y = np.array(x), np.array(y)
	random_shuffle_index = np.arange(x.shape[0])
	np.random.shuffle(random_shuffle_index)
	x = x[random_shuffle_index]
	y = y[random_shuffle_index]
	return x, y

def plot_graph(x, y, name='graph.png'):
	indexes = np.array([i for i in range(0, x.shape[0])])
	x_plot, y_plot = x[indexes], y[indexes]
	reds = np.array([i for i,j in zip(x_plot, y_plot) if j==1]) 
	blues = np.array([i for i,j in zip(x_plot, y_plot) if j==0]) 
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=reds.T[0], y=reds.T[1], mode='markers',name='Red: 1',marker=dict(color='Red')))
	fig.add_trace(go.Scatter(x=blues.T[0], y=blues.T[1], mode='markers',name='Blue: 0',marker=dict(color='Blue')))
	fig.add_trace(go.Scatter(x=[np.min(x.T[0]), np.max(x.T[0])], y=[0, 0], mode='lines',name='X-axis',marker=dict(color='Black')))
	fig.add_trace(go.Scatter(x=[0, 0], y=[-1, 1], mode='lines',name='Y-axis',marker=dict(color='Black')))
	st.plotly_chart(fig)

def generate_model(w=None, depth=1):
	model = [
		tf.keras.layers.Input(2),
		tf.keras.layers.Dense(3, activation='relu'),
		tf.keras.layers.Dense(6, activation='relu'),
		]
	for _ in range(depth):
		model.append(tf.keras.layers.Dense(3, activation='relu'))
	model.append(tf.keras.layers.Dense(1, activation='sigmoid'))
	model = tf.keras.Sequential(model)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
	if w!=None:
		model.set_weights(w)
	return model

def ann_learner(x, y):
	model = generate_model()
	history = model.fit(x, y, epochs=50, validation_split=0.1, verbose=False)
	return history.history

def federated_learner(x, y, n=2, depth=1):
	global client_epochs, aggregation_epochs
	textPlaceholder = st.empty()
	textPlaceholder.text("Beginning FL Training...")
	central_model = generate_model(depth=depth)
	central_w = None
	history = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
	for main_epoch in range(aggregation_epochs):
		weights = []
		for i in range(n):
			textPlaceholder.text("Beginning Client_"+ str(i+1) +" Training...   for main_epoch="+str(main_epoch+1))
			m = generate_model(central_w, depth=depth)
			m.fit(x[int(i*(x.shape[0]/n)):int((i+1)*(x.shape[0]/n))], y[int(i*(x.shape[0]/n)):int((i+1)*(x.shape[0]/n))], epochs=client_epochs, validation_split=0.1, verbose=False)
			weights.append(m.get_weights())
			textPlaceholder.text("Completed Client_"+ str(i+1) +" Training.\tfor main_epoch="+str(main_epoch+1))
		textPlaceholder.text("Now aggregating... for main_epoch="+str(main_epoch+1))
		weights = np.array(weights)
		weights = np.mean(weights, axis=0)
		central_w = weights
		central_model.set_weights(central_w)
		l, a = central_model.evaluate(x[:int(0.1*x.shape[0])], y[:int(0.1*x.shape[0])], verbose=False)
		vl, va = central_model.evaluate(x[-int(0.1*x.shape[0]):], y[-int(0.1*x.shape[0]):], verbose=False)
		history['loss'].append(l)
		history['acc'].append(a)
		history['val_acc'].append(va)
		history['val_loss'].append(vl)
		textPlaceholder.text("Aggregation Complete.")
	return history

def training_plot(history):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=[i for i in range(len(history['loss']))], y=history['loss'],mode='lines+markers',name='Loss'))
	fig.add_trace(go.Scatter(x=[i for i in range(len(history['val_loss']))], y=history['val_loss'],mode='lines+markers',name='Validation Loss'))
	fig.update_layout(yaxis=dict(range=[0,1]))
	st.plotly_chart(fig)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=[i for i in range(len(history['acc']))], y=history['acc'],mode='lines+markers',name='Accuracy'))
	fig.add_trace(go.Scatter(x=[i for i in range(len(history['val_acc']))], y=history['val_acc'],mode='lines+markers',name='Validation Accuracy'))
	fig.update_layout(yaxis=dict(range=[0,1]))
	st.plotly_chart(fig)

	h = pd.DataFrame(history)
	st.dataframe(h.tail())


def build_app():
	st.title("ANN v/s Federated Learning")
	st.markdown("Federated learning is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. It allows for Privacy Enabled Machine Learnign implementation.")
	st.header("DATASET:")
	st.markdown("To demonstrate such a comparison we will be generating a simple dataset, consisting of points above and below the Sine Curve. The aim of the ANN would be to classify those above the Sine Curve as 1 and those below as 0s. Take a look at the Generated Dataset below:")
	x, y = get_data()
	plot_graph(x, y)
	sample = generate_model()
	st.header("Simple Neural Networks")
	st.markdown("Let us look at a Simple ANN model learning the above function.")
	x, y = get_data()
	description = {"Number of epochs": 50, "Data Sample size": x.shape[0]}
	st.write(description)
	h_ann = ann_learner(x, y)
	training_plot(h_ann)
	st.header("Federated Learning")
	st.markdown("Having seen the performance of a simple ANN, let us take a look at Federated Learning results:")
	
	value = st.sidebar.slider("Number of Clients", 2, 5)
	depth = st.sidebar.slider("Depth of Network", 1, 5)
	description = {"Number of Clients": value, "Data Sample per Client":x.shape[0]//value, "Number of Training epochs per Client before Aggregation":client_epochs, "Number of time to Aggregate the model": aggregation_epochs}
	st.write(description)
	h_fl = federated_learner(x, y, n=value, depth=depth)
	training_plot(h_fl)

	st.markdown("Feel free to play around with FL model, by changing parameters using the slider option. However, there are only 4,500 Data Samples in total and each Client receives an equal and unique part of it. Therefore, be careful with the number of clients.")

if __name__ == '__main__':
	build_app()