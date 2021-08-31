# """ Projet de Recherche Operationnelle """



# en premier lieu nous allons importer les librairies
import numpy as np
import pandas as pd
import streamlit as st
import plotly_express as px
import time
import os
import streamlit.components.v1 as components
import string
import pygraphviz
import sys
from io import BytesIO,StringIO
import networkx as nx
from IPython.core.display import display,HTML
from pyvis import network as net
from pathlib import Path
import pandas as pd
import streamlit as st

# Titre de la page
st.write("""
        # Plateforme de Recherche Opérationnelle 
        ##
        """)

# algorithme de djikstra
def Djikstra(mg, s):
    st.write("""
            #
            ## Résultat de l'algorithme de Djikstra :
            """)
    infini = sum(sum(ligne) for ligne in mg)+1
    nb_sommets = len(mg)

    s_connu = {s: [0,[s]]}
    s_inconnu = {k: [infini, ''] for k in range(nb_sommets) if k !=s}

    for suivant in range(nb_sommets):
        if mg[s][suivant]:
            s_inconnu[suivant] = [mg[s][suivant], s]
    st.write("Dans le graphe d'origine {} de matrice d'adjacence:".format(s))
    for ligne in mg:
        print(ligne)
    st.write()
    st.write('Les Plus courts chemins de')

    while s_inconnu and any(s_inconnu[k][0] < infini for k in s_inconnu):
        u = min(s_inconnu, key= s_inconnu.get)
        longueur_u, precedent_u = s_inconnu[u]
        for v in range(nb_sommets):
            if mg[u][v] and v in s_inconnu:
                d = longueur_u + mg[u][v]
                if d < s_inconnu[v][0]:
                    s_inconnu[v] = [d, u]
        s_connu[u] = [longueur_u, s_connu[precedent_u][1] + [u]]
        del s_inconnu[u]
        st.write("longueur", longueur_u, ":", "->".join(map(str, s_connu[u][1])))
    for k in s_inconnu:
        st.write("Il n'y a aucun chemin de {} à {}.".format(s,k))

    return s_connu

# Algorithme de Prim
def Prim(G):
    st.write("""
                # 
                ## Résultat de l'algorithme de Prim  :
                """)
    INF = 999999999
    V = len(G)
    mincost=0
    selected = [0]*len(G)
    no_edge = 0
    selected[0] = True
    st.write("Liaison : Poids\n")
    while (no_edge < V - 1):
        minimum = INF
        x = 0
        y = 0
        for i in range(V):
            if selected[i]:
                for j in range(V):
                    if ((not selected[j]) and G[i][j]):
                        # not in selected and there is an edge
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            x = i
                            y = j
        mincost += minimum
        st.write(str(x) + " - " + str(y) + " : " + str(G[x][y]))
        selected[y] = True
        no_edge += 1
    st.write("Coût minimale = {}".format(mincost))

# Algorithme de Kruskal
def Kruskal(Matrice):
    # Find set of vertex i
    st.write("""
                    # 
                    ## Résultat de l'algorithme de Kruskal  :
                    """)
    V = len(Matrice)
    mincost = 0  # Cost of min MST
    parent = [i for i in range(V)]
    INF = 99999999

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    # Does union of i and j. It returns
    # false if i and j are already in same
    # set.
    def union(i, j):
        a = find(i)
        b = find(j)
        parent[a] = b


    # Initialize sets of disjoint sets
    for i in range(V):
        parent[i] = i

    # Include minimum weight edges one by one
    edge_count = 0
    while edge_count < V - 1:
        min = INF
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if Matrice[i][j] != 0 and find(i) != find(j) and Matrice[i][j] < min:
                    min = Matrice[i][j]
                    a = i
                    b = j
        union(a, b)
        st.write('Lien ({}, {}) coût : {}'.format(a, b, min))
        edge_count += 1
        mincost += min

    st.write("Coût Minimal = {}".format(mincost))

# Algorithme de Ford-fulkerson
def ford_fulkerson(graph):
    st.write("""
                        # 
                        ## Resultat de l'Algorithme de Ford-Fulkerson :
                        #### on a chercher à determiner le flot maximal  et l'algorithme nous retourne que :
                        # 
                        """)
    source = 0
    sink = len(graph) - 1

    # Using BFS as a searching algorithm
    def searching_algo_BFS(s, t, parent):

        visited = [False] * (len(graph))
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    parent = [-1] * (len(graph))
    max_flow = 0

    while searching_algo_BFS(source, sink, parent):

        path_flow = float("Inf")
        s = sink
        while (s != source):
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]

        # Adding the path flows
        max_flow += path_flow

        # Updating the residual values of edges
        v = sink
        while (v != source):
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]

    return max_flow


# Fonction Principale
def main():
    # on crée une bar à gauche d'ou on mettra des boutons pour nos différents fonctionnalités
    st.sidebar.write("""
            
            ## Importer la matrice d'adjacence
            
            """)

    # on demande à l'utilisateur d'importer un fichier de type CSV
    file = st.sidebar.file_uploader("",type=["csv"])
    # on défini une case vide ou on va mettre l'information montrant à l'utilisateur qu'il n'y a pas de fichier
    show_file = st.sidebar.empty()

    # on verifie s'il  n'y a aucun fichier importer
    if not file:
        # on lui renvoie l'info de charger la matrice d'adjacence
        show_file.info("Veuillez charger la matrice noeud noeud pour voir apparaitre toute les fonctionnalités")
        return

    # df est la variable qui va contenir le fichier importer
    df = pd.read_csv(file, header=None)
    # df1 la variable qui va contenir la conversion de df en un tableau numpy
    df1 = np.array(df)
    taille = len(df1)
    st.write("Ceci est le contenu de votre fichier CSV correspondant à votre matrice d'adjacence.")
    # affichage du dataframe charger convertit
    st.dataframe(df1)

    # fermer le fichier
    file.close()

    st.sidebar.write("""
                        ### Visualiser le graphe:
            """)
    # checkbox permettant de visualiser le graphe
    visualiser = st.sidebar.checkbox("Visualiser le Graphe")

    st.sidebar.write("""

                ## Appliquer un Algorithme 
                ###
                ### Plus court chemin : 

                """)

    # boutton permettant d'appliquer l'algorithme de djikstra
    djikstra = st.sidebar.button("Algorithme de Djikstra")
    # l'appui sur le button fait appel à la fonction implémentant l'algorithme de djikstra
    if djikstra:
        Djikstra(df1, 0)





    st.sidebar.write("""
                    ###
                    ### Arbre couvrant minimum : 

                    """)
    # boutton permettant d'appliquer l'algorithme de Prim
    algoPrim = st.sidebar.button("Algorithme de Prim")
    # l'appui sur le button fait appel à la fonction implémentant l'algorithme de Prim
    if algoPrim:
        Prim(df1)

    # boutton permettant d'appliquer l'algorithme de Kruskal
    algoKruskal = st.sidebar.button("Algorithme de Kruskal")
    # l'appui sur le button fait appel à la fonction implémentant l'algorithme de Kruskal
    if algoKruskal:
        Kruskal(df1)

    st.sidebar.write("""
                        ###
                        ### Flot Maximum : 

                        """)
    # boutton permettant d'appliquer l'algorithme de Ford_fulkerson
    algoFord_fulkerson = st.sidebar.button("Algorithme de Ford-Fulkerson")
    # l'appui sur le button fait appel à la fonction implémentant l'algorithme de Ford_fulkerson
    if algoFord_fulkerson:
        st.write("Flot Maximal = {} ".format(ford_fulkerson(df1)))

    # affichage du graphe si le checkbox est cocher
    if visualiser:
        st.write("""
                                # 
                                # Graphe de la matrice d'adjacence :
                                """)
        # definir le graphe en fonction de la matrice et de la librairie networkx
        G = nx.from_numpy_matrix(df1)
        # si on voulais afficher le graphe avec de noeud en lettre Alphabétiques on allait décommenter ce commentaire
        # G = nx.relabel_nodes(G, dict(zip(G, string.ascii_uppercase)))
        # on défini comment le graphe vas etre afficher
        g = net.Network(height='500px', width='100%', heading="")
        # cette methode prend un graph existant et le traduit en graph de format PyVis
        g.from_nx(G)
        # enregistrer le graphe
        g.save_graph("graphe.html")
        #ouvrir le graphe enregistrer dans un format html
        HtmlFile = open("graphe.html", 'r', encoding='utf-8')
        # lire le fichier html contenant le graphe
        souce_code = HtmlFile.read()
        components.html(souce_code, height=1000, width=1000)


# exécution de la fonction principale 
main()



