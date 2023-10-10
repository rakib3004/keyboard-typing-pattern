import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import joblib

data = pd.read_csv('pattern.csv')
X = data.drop(columns=['Target','user'])
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

if __name__ == "__main__":
    # Get the JSON string from the command-line arguments
    user_input = {'H.a': 0, 'DD.a.a': 0.1325, 'UD.a.a': 0, 'DD.a.b': 0, 'UD.a.b': 0.0, 'DD.a.c': 0.0, 'UD.a.c': 0.188, 'DD.a.d': 0.063, 'UD.a.d': 0, 'DD.a.e': 0, 'UD.a.e': 0, 'DD.a.f': 0, 'UD.a.f': 0, 'DD.a.g': 0, 'UD.a.g': 0, 'DD.a.h': 0, 'UD.a.h': 0, 'DD.a.i': 0, 'UD.a.i': 0.0, 'DD.a.j': 0.0, 'UD.a.j': 0, 'DD.a.k': 0, 'UD.a.k': 0, 'DD.a.l': 0, 'UD.a.l': 0.064, 'DD.a.m': 0.0, 
'UD.a.m': 0, 'DD.a.n': 0, 'UD.a.n': 0, 'DD.a.o': 0, 'UD.a.o': 0, 'DD.a.p': 0, 'UD.a.p': 0.0, 'DD.a.q': 0.0, 'UD.a.q': 0, 
'DD.a.r': 0, 'UD.a.r': 0.093, 'DD.a.s': -0.0675, 'UD.a.s': 0.0, 'DD.a.t': 0.0, 'UD.a.t': 0.104333333, 'DD.a.u': -0.064, 'UD.a.u': 0, 'DD.a.v': 0, 'UD.a.v': 0.0, 'DD.a.w': 0.0, 'UD.a.w': 0, 'DD.a.x': 0, 'UD.a.x': 0, 'DD.a.y': 0, 'UD.a.y': 0, 'DD.a.z': 0, 'UD.a.z': 0, 'DD.a.space': 0, 'UD.a.space': 0.237, 'H.b': 0.192, 'DD.b.a': 0.0, 'UD.b.a': 0, 'DD.b.b': 0, 'UD.b.b': 0, 'DD.b.c': 0, 'UD.b.c': 0, 'DD.b.d': 0, 'UD.b.d': 0, 'DD.b.e': 0, 'UD.b.e': 0, 'DD.b.f': 0, 'UD.b.f': 0, 'DD.b.g': 0, 'UD.b.g': 0, 'DD.b.h': 0, 'UD.b.h': 0, 'DD.b.i': 0, 'UD.b.i': 0, 'DD.b.j': 0, 'UD.b.j': 0, 'DD.b.k': 0, 'UD.b.k': 0, 'DD.b.l': 0, 'UD.b.l': 0.0, 'DD.b.m': 0.0, 'UD.b.m': 0, 'DD.b.n': 0, 'UD.b.n': 0, 'DD.b.o': 0, 'UD.b.o': 0, 'DD.b.p': 0, 'UD.b.p': 0, 'DD.b.q': 0, 'UD.b.q': 0, 'DD.b.r': 0, 'UD.b.r': 0, 'DD.b.s': 0, 'UD.b.s': 0, 'DD.b.t': 0, 'UD.b.t': 0, 'DD.b.u': 0, 'UD.b.u': 0, 'DD.b.v': 0, 'UD.b.v': 0, 'DD.b.w': 0, 'UD.b.w': 0, 'DD.b.x': 0, 'UD.b.x': 0, 'DD.b.y': 0, 'UD.b.y': 0, 'DD.b.z': 0, 'UD.b.z': 0, 'DD.b.space': 0, 'UD.b.space': 0, 'H.c': 0, 'DD.c.a': 0.1415, 'UD.c.a': 0, 'DD.c.b': 0, 
'UD.c.b': 0, 'DD.c.c': 0, 'UD.c.c': 0.0, 'DD.c.d': 0.0, 'UD.c.d': 0, 'DD.c.e': 0, 'UD.c.e': 0, 'DD.c.f': 0, 'UD.c.f': 0, 
'DD.c.g': 0, 'UD.c.g': 0, 'DD.c.h': 0, 'UD.c.h': 0.251, 'DD.c.i': 0.122, 'UD.c.i': 0.0, 'DD.c.j': 0.0, 'UD.c.j': 0, 'DD.c.k': 0, 'UD.c.k': 0, 'DD.c.l': 0, 'UD.c.l': 0.186, 'DD.c.m': 0.125, 'UD.c.m': 0, 'DD.c.n': 0, 'UD.c.n': 0, 'DD.c.o': 0, 'UD.c.o': 0.183, 'DD.c.p': 0.06, 'UD.c.p': 0, 'DD.c.q': 0, 'UD.c.q': 0, 'DD.c.r': 0, 'UD.c.r': 0, 'DD.c.s': 0, 'UD.c.s': 0.253, 'DD.c.t': 0.0, 'UD.c.t': 0.0, 'DD.c.u': 0.0, 'UD.c.u': 0, 'DD.c.v': 0, 'UD.c.v': 0, 'DD.c.w': 0, 'UD.c.w': 0, 'DD.c.x': 0, 'UD.c.x': 0, 'DD.c.y': 0, 'UD.c.y': 0, 'DD.c.z': 0, 'UD.c.z': 0, 'DD.c.space': 0, 'UD.c.space': 0, 'H.d': 0, 'DD.d.a': 0.0635, 'UD.d.a': 0, 'DD.d.b': 0, 'UD.d.b': 0, 'DD.d.c': 0, 'UD.d.c': 0, 'DD.d.d': 0, 'UD.d.d': 0, 'DD.d.e': 0, 'UD.d.e': 0.269, 'DD.d.f': 0.198, 'UD.d.f': 0, 'DD.d.g': 0, 'UD.d.g': 0, 'DD.d.h': 0, 'UD.d.h': 0, 'DD.d.i': 0, 'UD.d.i': 0.0, 'DD.d.j': 0.0, 'UD.d.j': 0, 'DD.d.k': 0, 'UD.d.k': 0, 'DD.d.l': 0, 'UD.d.l': 0, 'DD.d.m': 0, 'UD.d.m': 0, 'DD.d.n': 0, 'UD.d.n': 0, 'DD.d.o': 0, 'UD.d.o': 0, 'DD.d.p': 0, 'UD.d.p': 0, 'DD.d.q': 0, 'UD.d.q': 0, 'DD.d.r': 0, 'UD.d.r': 0, 'DD.d.s': 0, 'UD.d.s': 0, 'DD.d.t': 0, 'UD.d.t': 0, 'DD.d.u': 0, 'UD.d.u': 0, 'DD.d.v': 0, 'UD.d.v': 0, 'DD.d.w': 0, 'UD.d.w': 0, 'DD.d.x': 0, 'UD.d.x': 0, 'DD.d.y': 0, 'UD.d.y': 0, 'DD.d.z': 0, 'UD.d.z': 0, 'DD.d.space': 0, 'UD.d.space': 0.119, 'H.e': 0.063, 'DD.e.a': 0.101833333, 'UD.e.a': 0.1545, 'DD.e.b': 0.061, 'UD.e.b': 0, 'DD.e.c': 0, 'UD.e.c': 0, 'DD.e.d': 0, 'UD.e.d': 0.186, 'DD.e.e': 0.069, 'UD.e.e': 0, 'DD.e.f': 0, 'UD.e.f': 0, 'DD.e.g': 0, 'UD.e.g': 0, 'DD.e.h': 0, 'UD.e.h': 0, 'DD.e.i': 0, 'UD.e.i': 0, 'DD.e.j': 0, 'UD.e.j': 0, 'DD.e.k': 0, 'UD.e.k': 0, 'DD.e.l': 0, 'UD.e.l': 0.187, 'DD.e.m': 0.0, 'UD.e.m': 0.0, 'DD.e.n': 0.0, 'UD.e.n': 0.122, 'DD.e.o': 0.065, 'UD.e.o': 0, 'DD.e.p': 0, 'UD.e.p': 0, 'DD.e.q': 0, 'UD.e.q': 0, 'DD.e.r': 0, 'UD.e.r': 0.0, 'DD.e.s': 0.0, 'UD.e.s': 0.0, 'DD.e.t': 0.0, 'UD.e.t': 0, 'DD.e.u': 0, 'UD.e.u': 0, 'DD.e.v': 0, 'UD.e.v': 0, 'DD.e.w': 0, 'UD.e.w': 0, 'DD.e.x': 0, 'UD.e.x': 0, 'DD.e.y': 0, 'UD.e.y': 0, 'DD.e.z': 0, 'UD.e.z': 0, 'DD.e.space': 0, 'UD.e.space': 0.063, 'H.f': 0.0, 'DD.f.a': 0.057, 'UD.f.a': 0.0, 'DD.f.b': 0.0, 'UD.f.b': 0, 'DD.f.c': 0, 'UD.f.c': 0, 'DD.f.d': 0, 'UD.f.d': 0, 'DD.f.e': 0, 'UD.f.e': 0, 'DD.f.f': 0, 'UD.f.f': 0, 'DD.f.g': 
0, 'UD.f.g': 0, 'DD.f.h': 0, 'UD.f.h': 0, 'DD.f.i': 0, 'UD.f.i': 0, 'DD.f.j': 0, 'UD.f.j': 0, 'DD.f.k': 0, 'UD.f.k': 0, 'DD.f.l': 0, 'UD.f.l': 0, 'DD.f.m': 0, 'UD.f.m': 0, 'DD.f.n': 0, 'UD.f.n': 0, 'DD.f.o': 0, 'UD.f.o': 0, 'DD.f.p': 0, 'UD.f.p': 0, 'DD.f.q': 0, 'UD.f.q': 0, 'DD.f.r': 0, 'UD.f.r': 0, 'DD.f.s': 0, 'UD.f.s': 0, 'DD.f.t': 0, 'UD.f.t': 0, 'DD.f.u': 0, 'UD.f.u': 0.0, 'DD.f.v': 0.0, 'UD.f.v': 0, 'DD.f.w': 0, 'UD.f.w': 0, 'DD.f.x': 0, 'UD.f.x': 0, 'DD.f.y': 0, 'UD.f.y': 0, 'DD.f.z': 0, 'UD.f.z': 0, 'DD.f.space': 0, 'UD.f.space': 0.12, 'H.g': 0.063, 'DD.g.a': 0.001, 'UD.g.a': 0, 'DD.g.b': 
0, 'UD.g.b': 0, 'DD.g.c': 0, 'UD.g.c': 0, 'DD.g.d': 0, 'UD.g.d': 0, 'DD.g.e': 0, 'UD.g.e': 0, 'DD.g.f': 0, 'UD.g.f': 0, 'DD.g.g': 0, 'UD.g.g': 0, 'DD.g.h': 0, 'UD.g.h': 0, 'DD.g.i': 0, 'UD.g.i': 0, 'DD.g.j': 0, 'UD.g.j': 0, 'DD.g.k': 0, 'UD.g.k': 0, 'DD.g.l': 0, 'UD.g.l': 0, 'DD.g.m': 0, 'UD.g.m': 0, 'DD.g.n': 0, 'UD.g.n': 0, 'DD.g.o': 0, 'UD.g.o': 0.0, 'DD.g.p': 0.0, 'UD.g.p': 0, 'DD.g.q': 0, 'UD.g.q': 0, 'DD.g.r': 0, 'UD.g.r': 0, 'DD.g.s': 0, 'UD.g.s': 0, 'DD.g.t': 0, 'UD.g.t': 0, 'DD.g.u': 0, 'UD.g.u': 0, 'DD.g.v': 0, 'UD.g.v': 0, 'DD.g.w': 0, 'UD.g.w': 0, 'DD.g.x': 0, 'UD.g.x': 0, 'DD.g.y': 0, 
'UD.g.y': 0, 'DD.g.z': 0, 'UD.g.z': 0, 'DD.g.space': 0, 'UD.g.space': 0.057, 'H.h': 0.056, 'DD.h.a': 0.065, 'UD.h.a': 0, 
'DD.h.b': 0, 'UD.h.b': 0, 'DD.h.c': 0, 'UD.h.c': 0, 'DD.h.d': 0, 'UD.h.d': 0, 'DD.h.e': 0, 'UD.h.e': 0.0, 'DD.h.f': 0.0, 
'UD.h.f': 0, 'DD.h.g': 0, 'UD.h.g': 0, 'DD.h.h': 0, 'UD.h.h': 0, 'DD.h.i': 0, 'UD.h.i': 0.0, 'DD.h.j': -0.065, 'UD.h.j': 
0, 'DD.h.k': 0, 'UD.h.k': 0, 'DD.h.l': 0, 'UD.h.l': 0, 'DD.h.m': 0, 'UD.h.m': 0.0, 'DD.h.n': 0.0, 'UD.h.n': 0, 'DD.h.o': 
0, 'UD.h.o': 0, 'DD.h.p': 0, 'UD.h.p': 0, 'DD.h.q': 0, 'UD.h.q': 0, 'DD.h.r': 0, 'UD.h.r': 0, 'DD.h.s': 0, 'UD.h.s': 0, 'DD.h.t': 0, 'UD.h.t': 0, 'DD.h.u': 0, 'UD.h.u': 0, 'DD.h.v': 0, 'UD.h.v': 0, 'DD.h.w': 0, 'UD.h.w': 0, 'DD.h.x': 0, 'UD.h.x': 0, 'DD.h.y': 0, 'UD.h.y': 0, 'DD.h.z': 0, 'UD.h.z': 0, 'DD.h.space': 0, 'UD.h.space': 0.0, 'H.i': 0.0, 'DD.i.a': 0.145, 'UD.i.a': 0, 'DD.i.b': 0, 'UD.i.b': 0, 'DD.i.c': 0, 'UD.i.c': 0.243, 'DD.i.d': 0.124, 'UD.i.d': 0, 'DD.i.e': 0, 'UD.i.e': 0, 'DD.i.f': 0, 'UD.i.f': 0, 'DD.i.g': 0, 'UD.i.g': 0, 'DD.i.h': 0, 'UD.i.h': 0, 'DD.i.i': 0, 'UD.i.i': 0, 'DD.i.j': 0, 'UD.i.j': 0, 'DD.i.k': 0, 'UD.i.k': 0, 'DD.i.l': 0, 'UD.i.l': 0.0, 'DD.i.m': 0.0, 'UD.i.m': 0, 'DD.i.n': 0, 'UD.i.n': 0.095, 'DD.i.o': -0.0935, 'UD.i.o': 0.13, 'DD.i.p': -0.057, 'UD.i.p': 0.0, 'DD.i.q': 0.0, 'UD.i.q': 0, 'DD.i.r': 0, 'UD.i.r': 0, 'DD.i.s': 0, 'UD.i.s': 0.186, 'DD.i.t': 0.0925, 'UD.i.t': 0.0, 'DD.i.u': 0.0, 'UD.i.u': 0, 'DD.i.v': 0, 'UD.i.v': 0, 'DD.i.w': 0, 'UD.i.w': 0, 'DD.i.x': 0, 'UD.i.x': 0, 'DD.i.y': 0, 'UD.i.y': 0, 'DD.i.z': 0, 'UD.i.z': 0, 'DD.i.space': 0, 'UD.i.space': 0, 'H.j': 0, 'DD.j.a': 0, 'UD.j.a': 0, 'DD.j.b': 0, 'UD.j.b': 0, 'DD.j.c': 0, 'UD.j.c': 0, 'DD.j.d': 0, 'UD.j.d': 0, 'DD.j.e': 0, 'UD.j.e': 0, 'DD.j.f': 0, 'UD.j.f': 0, 'DD.j.g': 0, 'UD.j.g': 0, 'DD.j.h': 0, 'UD.j.h': 0, 'DD.j.i': 0, 'UD.j.i': 0, 'DD.j.j': 0, 'UD.j.j': 0, 'DD.j.k': 0, 'UD.j.k': 0, 'DD.j.l': 0, 'UD.j.l': 0, 'DD.j.m': 0, 'UD.j.m': 0, 'DD.j.n': 0, 'UD.j.n': 0, 'DD.j.o': 0, 'UD.j.o': 0, 'DD.j.p': 0, 'UD.j.p': 0, 'DD.j.q': 0, 'UD.j.q': 0, 'DD.j.r': 
0, 'UD.j.r': 0, 'DD.j.s': 0, 'UD.j.s': 0, 'DD.j.t': 0, 'UD.j.t': 0, 'DD.j.u': 0, 'UD.j.u': 0, 'DD.j.v': 0, 'UD.j.v': 0, 'DD.j.w': 0, 'UD.j.w': 0, 'DD.j.x': 0, 'UD.j.x': 0, 'DD.j.y': 0, 'UD.j.y': 0, 'DD.j.z': 0, 'UD.j.z': 0, 'DD.j.space': 0, 'UD.j.space': 0, 'H.k': 0, 'DD.k.a': 0.0, 'UD.k.a': 0, 'DD.k.b': 0, 'UD.k.b': 0, 'DD.k.c': 0, 'UD.k.c': 0, 'DD.k.d': 0, 'UD.k.d': 0, 'DD.k.e': 0, 'UD.k.e': 0, 'DD.k.f': 0, 'UD.k.f': 0, 'DD.k.g': 0, 'UD.k.g': 0, 'DD.k.h': 0, 'UD.k.h': 0, 'DD.k.i': 0, 'UD.k.i': 0, 'DD.k.j': 0, 'UD.k.j': 0, 'DD.k.k': 0, 'UD.k.k': 0, 'DD.k.l': 0, 'UD.k.l': 0, 'DD.k.m': 0, 'UD.k.m': 
0, 'DD.k.n': 0, 'UD.k.n': 0, 'DD.k.o': 0, 'UD.k.o': 0, 'DD.k.p': 0, 'UD.k.p': 0, 'DD.k.q': 0, 'UD.k.q': 0, 'DD.k.r': 0, 'UD.k.r': 0, 'DD.k.s': 0, 'UD.k.s': 0.0, 'DD.k.t': 0.0, 'UD.k.t': 0, 'DD.k.u': 0, 'UD.k.u': 0, 'DD.k.v': 0, 'UD.k.v': 0, 'DD.k.w': 0, 'UD.k.w': 0, 'DD.k.x': 0, 'UD.k.x': 0, 'DD.k.y': 0, 'UD.k.y': 0, 'DD.k.z': 0, 'UD.k.z': 0, 'DD.k.space': 0, 'UD.k.space': 0, 'H.l': 0, 'DD.l.a': 0.1264, 'UD.l.a': 0.126, 'DD.l.b': 0.0, 'UD.l.b': 0, 'DD.l.c': 0, 'UD.l.c': 0, 'DD.l.d': 0, 'UD.l.d': 0, 'DD.l.e': 0, 'UD.l.e': 0.1595, 'DD.l.f': 0.0635, 'UD.l.f': 0, 'DD.l.g': 0, 'UD.l.g': 0.0, 'DD.l.h': 0.0, 'UD.l.h': 0, 'DD.l.i': 0, 'UD.l.i': 0.0, 'DD.l.j': 0.0, 'UD.l.j': 0, 'DD.l.k': 0, 'UD.l.k': 0, 'DD.l.l': 0, 'UD.l.l': 0.0, 'DD.l.m': 0.0, 'UD.l.m': 0, 'DD.l.n': 0, 'UD.l.n': 0, 'DD.l.o': 0, 'UD.l.o': 0.0, 'DD.l.p': 0.0, 'UD.l.p': 0, 'DD.l.q': 0, 'UD.l.q': 0, 'DD.l.r': 0, 'UD.l.r': 0, 'DD.l.s': 0, 'UD.l.s': 0, 'DD.l.t': 0, 'UD.l.t': 0, 'DD.l.u': 0, 'UD.l.u': 0, 'DD.l.v': 0, 'UD.l.v': 0, 'DD.l.w': 0, 'UD.l.w': 0, 'DD.l.x': 0, 'UD.l.x': 0, 'DD.l.y': 0, 'UD.l.y': 0.189, 'DD.l.z': 0.063, 'UD.l.z': 0, 'DD.l.space': 0, 'UD.l.space': 0.253, 'H.m': 0.065, 'DD.m.a': 0.032, 'UD.m.a': 0.124, 'DD.m.b': 0.123, 'UD.m.b': 0, 'DD.m.c': 0, 'UD.m.c': 0, 'DD.m.d': 0, 'UD.m.d': 0, 'DD.m.e': 0, 'UD.m.e': 0, 'DD.m.f': 0, 'UD.m.f': 0, 'DD.m.g': 0, 'UD.m.g': 0, 'DD.m.h': 0, 'UD.m.h': 0, 'DD.m.i': 0, 'UD.m.i': 0, 'DD.m.j': 0, 'UD.m.j': 0, 'DD.m.k': 0, 'UD.m.k': 0, 'DD.m.l': 0, 'UD.m.l': 0, 'DD.m.m': 0, 'UD.m.m': 0, 'DD.m.n': 0, 'UD.m.n': 0, 'DD.m.o': 0, 'UD.m.o': 0, 'DD.m.p': 0, 'UD.m.p': 0.119, 'DD.m.q': 0.056, 'UD.m.q': 0, 'DD.m.r': 0, 'UD.m.r': 0, 'DD.m.s': 0, 'UD.m.s': 0, 'DD.m.t': 0, 'UD.m.t': 0, 'DD.m.u': 0, 'UD.m.u': 0, 'DD.m.v': 0, 'UD.m.v': 0, 'DD.m.w': 0, 'UD.m.w': 0, 'DD.m.x': 0, 'UD.m.x': 0, 'DD.m.y': 0, 'UD.m.y': 0, 'DD.m.z': 0, 'UD.m.z': 0, 'DD.m.space': 0, 'UD.m.space': 0.0, 'H.n': 0.0, 'DD.n.a': 0.0874, 'UD.n.a': 0.188, 'DD.n.b': 0.122, 'UD.n.b': 0, 'DD.n.c': 0, 'UD.n.c': 0, 'DD.n.d': 0, 'UD.n.d': 0, 'DD.n.e': 0, 'UD.n.e': 0.172, 'DD.n.f': 0.047, 'UD.n.f': 0, 'DD.n.g': 0, 'UD.n.g': 0.191, 'DD.n.h': 0.128, 'UD.n.h': 0, 'DD.n.i': 0, 'UD.n.i': 0.249, 'DD.n.j': 0.186, 'UD.n.j': 0, 'DD.n.k': 0, 'UD.n.k': 0, 'DD.n.l': 0, 'UD.n.l': 0, 'DD.n.m': 0, 'UD.n.m': 0, 'DD.n.n': 0, 'UD.n.n': 0, 'DD.n.o': 0, 'UD.n.o': 0.0, 'DD.n.p': 0.0, 'UD.n.p': 0, 'DD.n.q': 0, 'UD.n.q': 0, 'DD.n.r': 0, 'UD.n.r': 0, 'DD.n.s': 0, 'UD.n.s': 0, 'DD.n.t': 0, 'UD.n.t': 0.122, 'DD.n.u': 0.002, 'UD.n.u': 0, 'DD.n.v': 0, 'UD.n.v': 0, 'DD.n.w': 0, 'UD.n.w': 0, 'DD.n.x': 0, 'UD.n.x': 0, 'DD.n.y': 0, 'UD.n.y': 0, 'DD.n.z': 0, 'UD.n.z': 0, 'DD.n.space': 0, 'UD.n.space': 
0, 'H.o': 0, 'DD.o.a': 0.0775, 'UD.o.a': 0.0, 'DD.o.b': 0.0, 'UD.o.b': 0, 'DD.o.c': 0, 'UD.o.c': 0, 'DD.o.d': 0, 'UD.o.d': 0, 'DD.o.e': 0, 'UD.o.e': 0, 'DD.o.f': 0, 'UD.o.f': 0.194, 'DD.o.g': 0.125, 'UD.o.g': 0, 'DD.o.h': 0, 'UD.o.h': 0, 'DD.o.i': 0, 'UD.o.i': 0, 'DD.o.j': 0, 'UD.o.j': 0, 'DD.o.k': 0, 'UD.o.k': 0, 'DD.o.l': 0, 'UD.o.l': 0, 'DD.o.m': 0, 'UD.o.m': 0.193, 'DD.o.n': 0.13, 'UD.o.n': 0.057, 'DD.o.o': 0.0, 'UD.o.o': 0, 'DD.o.p': 0, 'UD.o.p': 0, 'DD.o.q': 0, 'UD.o.q': 0, 'DD.o.r': 0, 'UD.o.r': 0.0, 'DD.o.s': 0.0, 'UD.o.s': 0, 'DD.o.t': 0, 'UD.o.t': 0, 'DD.o.u': 0, 'UD.o.u': 0.0, 'DD.o.v': 
0.0, 'UD.o.v': 0, 'DD.o.w': 0, 'UD.o.w': 0, 'DD.o.x': 0, 'UD.o.x': 0, 'DD.o.y': 0, 'UD.o.y': 0.0, 'DD.o.z': 0.0, 'UD.o.z': 0, 'DD.o.space': 0, 'UD.o.space': 0.188, 'H.p': 0.067, 'DD.p.a': 0.253, 'UD.p.a': 0, 'DD.p.b': 0, 'UD.p.b': 0, 'DD.p.c': 0, 'UD.p.c': 0, 'DD.p.d': 0, 'UD.p.d': 0, 'DD.p.e': 0, 'UD.p.e': 0, 'DD.p.f': 0, 'UD.p.f': 0.0, 'DD.p.g': 0.0, 'UD.p.g': 0, 'DD.p.h': 0, 'UD.p.h': 0, 'DD.p.i': 0, 'UD.p.i': 0, 'DD.p.j': 0, 'UD.p.j': 0, 'DD.p.k': 0, 'UD.p.k': 0, 'DD.p.l': 0, 'UD.p.l': 0.0, 'DD.p.m': 0.0, 'UD.p.m': 0, 'DD.p.n': 0, 'UD.p.n': 0, 'DD.p.o': 0, 'UD.p.o': 0, 'DD.p.p': 0, 'UD.p.p': 0.0, 'DD.p.q': 0.0, 'UD.p.q': 0, 'DD.p.r': 0, 'UD.p.r': 0.0, 'DD.p.s': 0.0, 'UD.p.s': 0, 'DD.p.t': 0, 'UD.p.t': 0, 'DD.p.u': 0, 'UD.p.u': 0.252, 'DD.p.v': -0.001, 'UD.p.v': 0, 'DD.p.w': 0, 'UD.p.w': 0, 'DD.p.x': 0, 'UD.p.x': 0, 'DD.p.y': 0, 'UD.p.y': 0, 'DD.p.z': 0, 'UD.p.z': 0, 'DD.p.space': 0, 'UD.p.space': 0, 'H.q': 0, 'DD.q.a': 0, 'UD.q.a': 0, 'DD.q.b': 0, 'UD.q.b': 0, 'DD.q.c': 0, 'UD.q.c': 0, 'DD.q.d': 0, 'UD.q.d': 0, 'DD.q.e': 0, 'UD.q.e': 0, 'DD.q.f': 0, 'UD.q.f': 0, 'DD.q.g': 0, 'UD.q.g': 0, 'DD.q.h': 0, 'UD.q.h': 0, 'DD.q.i': 0, 'UD.q.i': 0, 'DD.q.j': 0, 'UD.q.j': 0, 'DD.q.k': 0, 'UD.q.k': 
0, 'DD.q.l': 0, 'UD.q.l': 0, 'DD.q.m': 0, 'UD.q.m': 0, 'DD.q.n': 0, 'UD.q.n': 0, 'DD.q.o': 0, 'UD.q.o': 0, 'DD.q.p': 0, 'UD.q.p': 0, 'DD.q.q': 0, 'UD.q.q': 0, 'DD.q.r': 0, 'UD.q.r': 0, 'DD.q.s': 0, 'UD.q.s': 0, 'DD.q.t': 0, 'UD.q.t': 0, 'DD.q.u': 0, 'UD.q.u': 0, 'DD.q.v': 0, 'UD.q.v': 0, 'DD.q.w': 0, 'UD.q.w': 0, 'DD.q.x': 0, 'UD.q.x': 0, 'DD.q.y': 0, 'UD.q.y': 0, 'DD.q.z': 0, 'UD.q.z': 0, 'DD.q.space': 0, 'UD.q.space': 0, 'H.r': 0, 'DD.r.a': 0.126333333, 'UD.r.a': 0, 'DD.r.b': 0, 'UD.r.b': 0, 'DD.r.c': 0, 'UD.r.c': 0, 'DD.r.d': 0, 'UD.r.d': 0, 'DD.r.e': 0, 'UD.r.e': 0.067, 'DD.r.f': -0.061, 'UD.r.f': 0, 'DD.r.g': 0, 'UD.r.g': 0, 'DD.r.h': 0, 'UD.r.h': 0, 'DD.r.i': 0, 'UD.r.i': 0.0, 'DD.r.j': 0.0, 'UD.r.j': 0, 'DD.r.k': 0, 'UD.r.k': 0, 'DD.r.l': 0, 'UD.r.l': 0.126, 'DD.r.m': 0.0, 'UD.r.m': 0, 'DD.r.n': 0, 'UD.r.n': 0.253, 'DD.r.o': 0.128, 'UD.r.o': 0.0, 'DD.r.p': 0.0, 'UD.r.p': 0, 'DD.r.q': 0, 'UD.r.q': 0, 'DD.r.r': 0, 'UD.r.r': 0, 'DD.r.s': 0, 'UD.r.s': 0.0, 'DD.r.t': 0.0, 'UD.r.t': 0, 'DD.r.u': 0, 'UD.r.u': 0, 'DD.r.v': 0, 'UD.r.v': 0, 'DD.r.w': 0, 'UD.r.w': 0, 'DD.r.x': 0, 'UD.r.x': 0, 'DD.r.y': 0, 'UD.r.y': 0.0, 'DD.r.z': 0.0, 'UD.r.z': 0, 'DD.r.space': 0, 'UD.r.space': 0, 'H.s': 0, 'DD.s.a': 0.0498, 'UD.s.a': 0.0, 'DD.s.b': 0.0, 'UD.s.b': 0, 'DD.s.c': 0, 'UD.s.c': 0.0, 'DD.s.d': 0.0, 'UD.s.d': 0, 'DD.s.e': 0, 'UD.s.e': 0, 'DD.s.f': 0, 'UD.s.f': 0.0, 'DD.s.g': 0.0, 'UD.s.g': 0, 'DD.s.h': 0, 'UD.s.h': 0.0, 'DD.s.i': 0.0, 'UD.s.i': 0, 'DD.s.j': 0, 'UD.s.j': 0, 'DD.s.k': 0, 'UD.s.k': 0.0, 'DD.s.l': 0.0, 'UD.s.l': 0, 'DD.s.m': 0, 'UD.s.m': 0, 'DD.s.n': 0, 'UD.s.n': 0, 'DD.s.o': 0, 'UD.s.o': 0, 'DD.s.p': 0, 'UD.s.p': 0, 'DD.s.q': 0, 'UD.s.q': 0, 'DD.s.r': 0, 'UD.s.r': 0, 'DD.s.s': 0, 'UD.s.s': 0, 'DD.s.t': 0, 'UD.s.t': 0.208666667, 'DD.s.u': 0.167333333, 'UD.s.u': 0, 'DD.s.v': 0, 'UD.s.v': 0, 'DD.s.w': 0, 'UD.s.w': 0, 'DD.s.x': 0, 'UD.s.x': 0, 'DD.s.y': 0, 'UD.s.y': 0, 'DD.s.z': 0, 'UD.s.z': 0, 'DD.s.space': 0, 'UD.s.space': 0.25, 'H.t': 0.183, 'DD.t.a': 0.101111111, 'UD.t.a': 0.123, 'DD.t.b': 0.0595, 'UD.t.b': 0, 'DD.t.c': 0, 'UD.t.c': 0, 'DD.t.d': 0, 'UD.t.d': 0, 'DD.t.e': 0, 'UD.t.e': 0.259, 'DD.t.f': 0.07, 'UD.t.f': 0, 'DD.t.g': 0, 'UD.t.g': 0, 'DD.t.h': 0, 'UD.t.h': 0.0, 'DD.t.i': 0.0, 'UD.t.i': 0.175333333, 'DD.t.j': 0.087, 'UD.t.j': 0, 'DD.t.k': 0, 'UD.t.k': 0, 'DD.t.l': 0, 'UD.t.l': 0, 'DD.t.m': 0, 'UD.t.m': 0, 'DD.t.n': 0, 'UD.t.n': 0, 'DD.t.o': 0, 'UD.t.o': 0.068, 'DD.t.p': 0.0, 'UD.t.p': 0, 'DD.t.q': 0, 'UD.t.q': 0, 'DD.t.r': 0, 'UD.t.r': 0, 'DD.t.s': 0, 'UD.t.s': 0, 'DD.t.t': 0, 'UD.t.t': 0, 'DD.t.u': 0, 'UD.t.u': 0.204, 'DD.t.v': 0.001, 'UD.t.v': 0, 'DD.t.w': 0, 'UD.t.w': 0, 'DD.t.x': 0, 'UD.t.x': 0, 'DD.t.y': 0, 'UD.t.y': 0, 'DD.t.z': 0, 'UD.t.z': 0, 'DD.t.space': 0, 'UD.t.space': 0.118, 'H.u': 0.06, 'DD.u.a': 0.0305, 'UD.u.a': 0, 'DD.u.b': 0, 'UD.u.b': 0, 'DD.u.c': 0, 'UD.u.c': 0, 'DD.u.d': 0, 'UD.u.d': 0.0, 'DD.u.e': 0.0, 'UD.u.e': 0, 'DD.u.f': 0, 'UD.u.f': 0, 'DD.u.g': 0, 'UD.u.g': 0, 'DD.u.h': 0, 'UD.u.h': 0, 'DD.u.i': 0, 'UD.u.i': 0, 'DD.u.j': 0, 'UD.u.j': 0, 'DD.u.k': 0, 'UD.u.k': 0, 'DD.u.l': 0, 'UD.u.l': 0.0, 'DD.u.m': 0.0, 'UD.u.m': 0, 'DD.u.n': 0, 'UD.u.n': 0, 'DD.u.o': 0, 'UD.u.o': 0, 'DD.u.p': 0, 'UD.u.p': 0, 'DD.u.q': 0, 'UD.u.q': 0, 'DD.u.r': 0, 'UD.u.r': 0, 'DD.u.s': 0, 'UD.u.s': 
0.0, 'DD.u.t': 0.0, 'UD.u.t': 0.254, 'DD.u.u': 0.193, 'UD.u.u': 0, 'DD.u.v': 0, 'UD.u.v': 0, 'DD.u.w': 0, 'UD.u.w': 0, 'DD.u.x': 0, 'UD.u.x': 0, 'DD.u.y': 0, 'UD.u.y': 0, 'DD.u.z': 0, 'UD.u.z': 0, 'DD.u.space': 0, 'UD.u.space': 0, 'H.v': 0, 'DD.v.a': 0.0, 'UD.v.a': 0.0, 'DD.v.b': 0.0, 'UD.v.b': 0, 'DD.v.c': 0, 'UD.v.c': 0, 'DD.v.d': 0, 'UD.v.d': 0, 'DD.v.e': 0, 'UD.v.e': 0, 'DD.v.f': 0, 'UD.v.f': 0, 'DD.v.g': 0, 'UD.v.g': 0, 'DD.v.h': 0, 'UD.v.h': 0, 'DD.v.i': 0, 'UD.v.i': 0, 'DD.v.j': 0, 'UD.v.j': 0, 'DD.v.k': 0, 'UD.v.k': 0, 'DD.v.l': 0, 'UD.v.l': 0, 'DD.v.m': 0, 'UD.v.m': 0, 'DD.v.n': 0, 'UD.v.n': 0, 'DD.v.o': 0, 'UD.v.o': 0, 'DD.v.p': 0, 'UD.v.p': 0, 'DD.v.q': 0, 'UD.v.q': 0, 'DD.v.r': 0, 'UD.v.r': 0, 'DD.v.s': 0, 'UD.v.s': 0, 'DD.v.t': 0, 'UD.v.t': 0, 'DD.v.u': 0, 'UD.v.u': 0, 'DD.v.v': 0, 'UD.v.v': 0, 'DD.v.w': 0, 'UD.v.w': 0, 'DD.v.x': 0, 'UD.v.x': 0, 'DD.v.y': 0, 'UD.v.y': 0, 'DD.v.z': 0, 'UD.v.z': 0, 'DD.v.space': 0, 'UD.v.space': 0, 'H.w': 0, 'DD.w.a': 0.0, 'UD.w.a': 0, 'DD.w.b': 0, 'UD.w.b': 0, 'DD.w.c': 0, 'UD.w.c': 0, 'DD.w.d': 0, 'UD.w.d': 0, 'DD.w.e': 0, 'UD.w.e': 0, 'DD.w.f': 0, 'UD.w.f': 0, 'DD.w.g': 0, 'UD.w.g': 0, 'DD.w.h': 0, 'UD.w.h': 0.0, 'DD.w.i': 0.0, 'UD.w.i': 0, 'DD.w.j': 0, 'UD.w.j': 0, 'DD.w.k': 0, 'UD.w.k': 0, 'DD.w.l': 0, 'UD.w.l': 0, 'DD.w.m': 0, 'UD.w.m': 0, 'DD.w.n': 0, 'UD.w.n': 0, 'DD.w.o': 0, 'UD.w.o': 0, 'DD.w.p': 0, 'UD.w.p': 0, 'DD.w.q': 0, 'UD.w.q': 0, 'DD.w.r': 0, 'UD.w.r': 0, 'DD.w.s': 0, 'UD.w.s': 0, 'DD.w.t': 0, 'UD.w.t': 0, 'DD.w.u': 0, 'UD.w.u': 0, 'DD.w.v': 0, 'UD.w.v': 0, 'DD.w.w': 0, 'UD.w.w': 0, 'DD.w.x': 0, 'UD.w.x': 0, 'DD.w.y': 0, 'UD.w.y': 0, 'DD.w.z': 0, 'UD.w.z': 0, 'DD.w.space': 0, 'UD.w.space': 0, 'H.x': 0, 'DD.x.a': 0, 'UD.x.a': 0, 'DD.x.b': 0, 'UD.x.b': 0, 'DD.x.c': 0, 'UD.x.c': 0, 'DD.x.d': 0, 'UD.x.d': 0, 'DD.x.e': 0, 'UD.x.e': 0, 'DD.x.f': 0, 'UD.x.f': 0, 'DD.x.g': 0, 'UD.x.g': 0, 'DD.x.h': 0, 'UD.x.h': 0, 'DD.x.i': 0, 'UD.x.i': 0, 'DD.x.j': 0, 'UD.x.j': 0, 'DD.x.k': 0, 'UD.x.k': 0, 'DD.x.l': 0, 'UD.x.l': 0, 'DD.x.m': 0, 'UD.x.m': 0, 'DD.x.n': 0, 'UD.x.n': 0, 
'DD.x.o': 0, 'UD.x.o': 0, 'DD.x.p': 0, 'UD.x.p': 0, 'DD.x.q': 0, 'UD.x.q': 0, 'DD.x.r': 0, 'UD.x.r': 0, 'DD.x.s': 0, 'UD.x.s': 0, 'DD.x.t': 0, 'UD.x.t': 0, 'DD.x.u': 0, 'UD.x.u': 0, 'DD.x.v': 0, 'UD.x.v': 0, 'DD.x.w': 0, 'UD.x.w': 0, 'DD.x.x': 0, 'UD.x.x': 0, 'DD.x.y': 0, 'UD.x.y': 0, 'DD.x.z': 0, 'UD.x.z': 0, 'DD.x.space': 0, 'UD.x.space': 0, 'H.y': 0, 'DD.y.a': 0.065, 'UD.y.a': 0, 'DD.y.b': 0, 'UD.y.b': 0, 'DD.y.c': 0, 'UD.y.c': 0, 'DD.y.d': 0, 'UD.y.d': 0, 'DD.y.e': 0, 'UD.y.e': 0.0, 'DD.y.f': 0.0, 'UD.y.f': 0, 'DD.y.g': 0, 'UD.y.g': 0, 'DD.y.h': 0, 'UD.y.h': 0, 'DD.y.i': 0, 'UD.y.i': 0, 'DD.y.j': 0, 'UD.y.j': 0, 'DD.y.k': 0, 'UD.y.k': 0, 'DD.y.l': 0, 'UD.y.l': 0, 'DD.y.m': 0, 'UD.y.m': 0, 'DD.y.n': 0, 'UD.y.n': 0, 'DD.y.o': 0, 'UD.y.o': 0, 'DD.y.p': 0, 'UD.y.p': 0, 'DD.y.q': 0, 'UD.y.q': 0, 'DD.y.r': 0, 'UD.y.r': 0, 'DD.y.s': 0, 'UD.y.s': 0, 'DD.y.t': 0, 'UD.y.t': 0, 'DD.y.u': 0, 'UD.y.u': 0, 'DD.y.v': 0, 'UD.y.v': 0, 'DD.y.w': 0, 'UD.y.w': 0, 'DD.y.x': 0, 'UD.y.x': 0, 'DD.y.y': 0, 'UD.y.y': 0, 'DD.y.z': 0, 'UD.y.z': 0, 'DD.y.space': 0, 'UD.y.space': 0.248, 'H.z': 0.183, 'DD.z.a': 0, 'UD.z.a': 0, 'DD.z.b': 0, 'UD.z.b': 0, 'DD.z.c': 0, 'UD.z.c': 0, 'DD.z.d': 0, 'UD.z.d': 0, 'DD.z.e': 0, 'UD.z.e': 0, 'DD.z.f': 0, 'UD.z.f': 0, 'DD.z.g': 0, 'UD.z.g': 0, 'DD.z.h': 0, 'UD.z.h': 0, 'DD.z.i': 0, 'UD.z.i': 0, 'DD.z.j': 0, 'UD.z.j': 0, 'DD.z.k': 0, 'UD.z.k': 0, 'DD.z.l': 0, 'UD.z.l': 0, 'DD.z.m': 0, 'UD.z.m': 0, 'DD.z.n': 0, 'UD.z.n': 0, 'DD.z.o': 0, 'UD.z.o': 0, 'DD.z.p': 0, 'UD.z.p': 0, 'DD.z.q': 0, 'UD.z.q': 0, 'DD.z.r': 0, 'UD.z.r': 0, 'DD.z.s': 0, 
'UD.z.s': 0, 'DD.z.t': 0, 'UD.z.t': 0, 'DD.z.u': 0, 'UD.z.u': 0, 'DD.z.v': 0, 'UD.z.v': 0, 'DD.z.w': 0, 'UD.z.w': 0, 'DD.z.x': 0, 'UD.z.x': 0, 'DD.z.y': 0, 'UD.z.y': 0, 'DD.z.z': 0, 'UD.z.z': 0, 'DD.z.space': 0, 'UD.z.space': 0, 'H.space': 0, 'DD.space.a': 0.1639, 'UD.space.a': 0.0, 'DD.space.b': 0.0, 'UD.space.b': 0, 'DD.space.c': 0, 'UD.space.c': 0.592, 'DD.space.d': 0.157, 'UD.space.d': 0.0, 'DD.space.e': 0.0, 'UD.space.e': 0.0, 'DD.space.f': 0.0, 'UD.space.f': 0.0, 'DD.space.g': 0.0, 'UD.space.g': 0, 'DD.space.h': 0, 'UD.space.h': 0, 'DD.space.i': 0, 'UD.space.i': 0.254, 'DD.space.j': 0.184, 'UD.space.j': 0, 'DD.space.k': 0, 'UD.space.k': 0, 'DD.space.l': 0, 'UD.space.l': 0.188, 'DD.space.m': 0.062, 'UD.space.m': 0.252, 'DD.space.n': 0.124, 'UD.space.n': 0.0, 'DD.space.o': 0.0, 'UD.space.o': 0.125, 'DD.space.p': 0.0, 'UD.space.p': 
0.0, 'DD.space.q': 0.0, 'UD.space.q': 0, 'DD.space.r': 0, 'UD.space.r': 0.189, 'DD.space.s': 0.121, 'UD.space.s': 1.9145, 'DD.space.t': 1.8195, 'UD.space.t': 0.186, 'DD.space.u': 0.124, 'UD.space.u': 0, 'DD.space.v': 0, 'UD.space.v': 0.0, 'DD.space.w': 0.0, 'UD.space.w': 0.0, 'DD.space.x': 0.0, 'UD.space.x': 0, 'DD.space.y': 0, 'UD.space.y': 0, 'DD.space.z': 0, 'UD.space.z': 0, 'DD.space.space': 0, 'UD.space.space': 0.0, 'Unnamed: 1486': 0.0}
    
    
user_data = pd.DataFrame([user_input])

prediction = model.predict(user_data)

result  = int(prediction[0])
print(json.dumps({"result": result}))

