import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt 
import seaborn as sns

import re


from scipy import interpolate
from math import log 
from math import e
from graphviz import Digraph
from graphviz import Source

temp = """
digraph G{
#	size ="100,100"
nodesep = 0.1
ranksep = 1.6;
	subgraph cluster0{
		node [shape = circle, style = filled, color = white, fontname = Helvetica,fixedsize=10]
	#	splines = line;
		 style = filled;
		color = gainsboro
		x6[label=<<FONT POINT-SIZE='20'>x</FONT><FONT POINT-SIZE='11'>n</FONT>>];
	        #x5[label=<<FONT POINT-SIZE='20'>x</FONT><FONT POINT-SIZE='11'>5</FONT>>];
		#x4[label=<<FONT POINT-SIZE='20'>x</FONT><FONT POINT-SIZE='11'>4</FONT>>];
		x5[label="..."];
		x4[label="..."];
		x3[label=<<FONT POINT-SIZE='20'>x</FONT><FONT POINT-SIZE='11'>3</FONT>>];
		x2[label=<<FONT POINT-SIZE='20'>x</FONT><FONT POINT-SIZE='11'>2</FONT>>];
		x1[label=<<FONT POINT-SIZE='20'>x</FONT><FONT POINT-SIZE='11'>1</FONT>>];

	}



	subgraph cluster1{
		node [shape = circle, style = filled, color = white, fontname = Helvetica,fixedsize=10]
	#	splines = line;
	 	style = filled;
		color = gainsboro
		y6[label="..."];
		y5[label="..."];
		y4[label="..."];
		y3[label=<<FONT POINT-SIZE='20'>z</FONT><FONT POINT-SIZE='11'>3</FONT>>];
		y2[label=<<FONT POINT-SIZE='20'>z</FONT><FONT POINT-SIZE='11'>2</FONT>>];
		y1[label=<<FONT POINT-SIZE='20'>z</FONT><FONT POINT-SIZE='11'>1</FONT>>];
		y9[label=<<FONT POINT-SIZE='20'>z</FONT><FONT POINT-SIZE='11'>n</FONT>>];
		y8[label="..."];
		y7[label="..."];
	}



	subgraph cluster2{
		node [shape = circle, style = filled, color = white, fontname = Helvetica,fixedsize=10]
		style = filled;
		color = gainsboro
		z2[label=<<FONT POINT-SIZE='20'>y</FONT><FONT POINT-SIZE='11'>2</FONT>>];
		z1[label=<<FONT POINT-SIZE='20'>y</FONT><FONT POINT-SIZE='11'>1</FONT>>];
		z4[label=<<FONT POINT-SIZE='20'>y</FONT><FONT POINT-SIZE='11'>n</FONT>>];
		z3[label="..."];
	}

	subgraph cluster3{
		node [label="",shape = circle,color=none,fontname = Helvetica,fixedsize=10]
		ff1;ff2;ff3;ff4;
		color=none
	}
	splines=line


	x1->{y1,y2,y3,y4,y5,y6,y7,y8,y9}[arrowhead = vee,arrowsize=0.5]
	x2->{y1,y2,y3,y4,y5,y6,y7,y8,y9}[arrowhead = vee,arrowsize=0.5]
	x3->{y1,y2,y3,y4,y5,y6,y7,y8,y9}[arrowhead = vee,arrowsize=0.5]
	x4->{y1,y2,y3,y4,y5,y6,y7,y8,y9}[arrowhead = vee,arrowsize=0.5]
	x5->{y1,y2,y3,y4,y5,y6,y7,y8,y9}[arrowhead = vee,arrowsize=0.5]
	x6->{y1,y2,y3,y4,y5,y6,y7,y8}[lable="1",arrowhead = vee,arrowsize=0.5]
        x6->y9[arrowhead = vee, arrowsize=0.5, lable="wn"]

	y1->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y2->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y3->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y4->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y5->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y6->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y7->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y8->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]
	y9->{z1,z2,z3,z4}[arrowhead = vee,arrowsize=0.5]



	z1->ff1[arrowhead = vee,arrowsize=0.5]
	z2->ff2[arrowhead = vee,arrowsize=0.5]
	z3->ff3[arrowhead = vee,arrowsize=0.5]
	z4->ff4[arrowhead = vee,arrowsize=0.5]
	



    AA[label="input layer",fontcolor = deepskyblue, fontname = Helvetica,fontsize=20,shape = box,color=none,fixedsize=false]
	BB[label="hidden layer(s)",fontcolor = deepskyblue,fontname = Helvetica,fontsize=20,shape = box,color=none,fixedsize=false]
	CC[label="output layer",fontcolor = deepskyblue,fontname = Helvetica,fontsize=20,shape = box,color=none,fixedsize=false]
	#DD[label="loss=f(y_pre,y_true)",fontcolor = deepskyblue,fontname = Helvetica,fontsize=20,shape = box,color=none,fixedsize=false]
	AA->BB->CC [color=none]
}
"""
plt.close("all")
s = Source(temp,filename="schematic_plot",format="eps")
s.view()
