import ast
from datetime import date
import lxml.etree as etree
import pandas as pd
import numpy as np
import os
import pickle
import time
import json_lines
import networkx as nx

from matplotlib import pyplot as plt

def deltai(G,n,i):

    Gr=G-eye(n,n)

    [deg,indeg,outdeg]=degrees(Gr)
    D=outdeg

    N=find(Gr(:,i))
    k=size(N,1)

    DOMI=[]

    for j=1:k
        DOMI(j)=D(N(j))

    domi=sum(DOMI)

    di=(D(i)^2)/domi

    return di

def hidden_opinions(n,E,Alpha,mu,tau,T):

    G=returnadj(E,n)

    #Step 2: recover the degrees
    [deg,indeg,outdeg]=degrees(G)
    D=outdeg

    #Step 3: comput local populairty parameter to determine Express/Hide
    delta=[]

    for i=1:n
        delta(i)=deltai(G,n,i)

    Expressers=find(delta>=1) #expressers
    Con=find(delta<1) #consensual

    #Step 4 process of interpersonal influence

    epsilon1=1-5*eps(1); # set tolerance because Matlab doesn't read 1.0000 as 1 or 0.9999999999999 as 1 (because let's face it, it's not!)
    epsilon2=-1+5*eps(-1);

    M=zeros(T,n); #Opinion matrix

    M(1,:)=transpose(Alpha)

    L=zeros(n,n) #matrix for the law of motion, to keep track of the opinions of expressers.

    for t=2:T :

        for i=Expressers :

            for j=Expressers :

                if G(i,j)==1 && abs(M(t-1,i)-M(t-1,j))>=tau && M(t-1,i)>epsilon2 && M(t-1,i)<epsilon1 && i~=j:

                    L(i,j)=mu*(M(t-1,i)-M(t-1,j))  #neigbors who repulse

                elif G(i,j)==1 && abs(M(t-1,i)-M(t-1,j))<tau && M(t-1,i)>epsilon2 && M(t-1,i)<epsilon1 && i~=j :

                    L(i,j)=mu*(M(t-1,j)-M(t-1,i))   #neighbors who attract


                M(t,i)=M(t-1,i) + sum(L(i,:))  #total effect/change/variation

                if M(t,i)<epsilon2:

                    M(t,i)=-1; #if M gets out of the bound then make it stop at the lower bound -1 !

                elif M(t,i)>epsilon1:

                    M(t,i)=1;  #if M gets out of the bound then make it stop at the upper bound 1 !

        for i=Con :

            M(t,i)=(G(i,:)*transpose(M(t-1,:)))/D(i);

         #break in case convergence before T
         ep=ones(1,n)*(10^10)*eps;
         dif=abs(M(t,:)-M(t-1,:))<ep ;

         if sum(dif)==size(dif,2):
             break

    #remove zero lines , if iteration breaks... if not then simply  K==M
    matrix_evolution=M(any(M,2),:)
    last_period=size(K,1)

    final_opinions=K(last_period,:)


    return Expressers, final_opinions, matrix_evolution, last_period

def plot_evolution_hidden_opinions():



def plot_graph_hidden_opinions():
