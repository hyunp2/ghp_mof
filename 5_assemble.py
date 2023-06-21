import os, io, random
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymatgen.core as mg

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def readPillaredPaddleWheelXYZ(fpath, dummyElementCOO="At", dummyElementPillar="Fr"):
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    COOAnchorIds = df[df["el"]==dummyElementCOO].index.tolist()
    PillarAnchorIds = df[df["el"]==dummyElementPillar].index.tolist()
    df["maxDist"] = 0.
    anchorPairs = []
    while len(COOAnchorIds) > 0:
        maxDist = 0
        currAnchor = COOAnchorIds[-1]
        COOAnchorIds = COOAnchorIds[:-1]
        df.loc[COOAnchorIds, "maxDist"] = np.linalg.norm(df.loc[currAnchor, ["x", "y", "z"]].astype(float).values - \
                                                         df.loc[COOAnchorIds, ["x", "y", "z"]].values, axis=1)

        pairAnchor = df[df["maxDist"] == df.loc[COOAnchorIds, "maxDist"].max()].index[0]
        COOAnchorIds = COOAnchorIds[:COOAnchorIds.index(pairAnchor)] + COOAnchorIds[COOAnchorIds.index(pairAnchor)+1:]
        anchorPairs.append([currAnchor, pairAnchor])
    return df[["el", "x", "y", "z"]], anchorPairs, PillarAnchorIds

def readTetramerXYZ(fpath, dummyElement="At"):
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    anchorIds = df[df["el"]==dummyElement].index.tolist()
    df["maxDist"] = 0.
    anchorPairs = []
    while len(anchorIds) > 0:
        maxDist = 0
        currAnchor = anchorIds[-1]
        anchorIds = anchorIds[:-1]
        df.loc[anchorIds, "maxDist"] = np.linalg.norm(df.loc[currAnchor, ["x", "y", "z"]].astype(float).values - \
                                                      df.loc[anchorIds, ["x", "y", "z"]].values, axis=1)

        pairAnchor = df[df["maxDist"] == df.loc[anchorIds, "maxDist"].max()].index[0]
        anchorIds = anchorIds[:anchorIds.index(pairAnchor)] + anchorIds[anchorIds.index(pairAnchor)+1:]
        anchorPairs.append([currAnchor, pairAnchor])
    
    return df[["el", "x", "y", "z"]], anchorPairs

def readLinkerXYZ(fpath, dummyElement="At"):
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    anchorIds = df[df["el"]==dummyElement].index.tolist()
    return df, anchorIds

def readNbasedLinkerXYZ(fpath, dummyElement="Fr"):
    df = pd.read_csv(fpath, sep=r"\s+", skiprows=2, names=["el", "x", "y", "z"])
    anchorIds = df[df["el"]==dummyElement].index.tolist()
    return df, anchorIds

def pandas2xyzfile(df, fpath):
    with io.open(fpath, "w", newline="\n") as wf:
        wf.write(str(len(df)) + "\n\n" + df.to_string(header=None, index=None))

def rotmat2align(vec1, vec2):
    # https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def assemble_Zn_Tetramer_pcuMOF_multiProc(inputDict):
    nodePath = inputDict["nodePath"]
    linkerPaths = inputDict["linkerPaths"]
    newMOFdir = inputDict["newMOFdir"]
    if "dummyElement" in inputDict.keys():
        dummyElement = inputDict["dummyElement"]
    else:
        dummyElement = "At"

    returnValue = None
    try:
        returnValue = assemble_Zn_tetramer_pcuMOF(nodePath, linkerPaths, newMOFdir, dummyElement=dummyElement)
    except Exception as e:
        print(e)
    return returnValue

def assemble_Zn_tetramer_pcuMOF(nodePath, linkerPaths, newMOFdir, dummyElement="At"):
    os.makedirs(newMOFdir, exist_ok=True)
    
    node_df, nodeAnchors = readTetramerXYZ(nodePath)
    linkerAnchors = [None for x in range(3)]
    linker_dfs = [None for x in range(3)]
    linker_dfs[0], linkerAnchors[0] = readLinkerXYZ(linkerPaths[0])
    linker_dfs[1], linkerAnchors[1] = readLinkerXYZ(linkerPaths[1])
    linker_dfs[2], linkerAnchors[2] = readLinkerXYZ(linkerPaths[2])
    ldf = []
    latVec = []
    for _i_ in range(len(nodeAnchors)):
        noderAnchorPair = nodeAnchors[_i_]
        nodeAnchor1POS = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values
        nodePaired2POS = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values
        anchorPairVec = nodeAnchor1POS - nodePaired2POS
        linker_df = linker_dfs[_i_]
        linkerAnchorPair = linkerAnchors[_i_]

        # translate to align n1 and l1
        linkerAnchor1POS = linker_df.loc[linkerAnchorPair[0], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS) * \
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS), 
                                    axis=1)
        linkerCarbon1Idx = linker_df[linker_df["dist2"]==linker_df[linker_df["el"]=="C"]["dist2"].min()].index.tolist()[0]

        linkerAnchor2POS = linker_df.loc[linkerAnchorPair[1], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS) * \
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS), 
                                    axis=1)
        linkerCarbon2Idx = linker_df[linker_df["dist2"]==linker_df[linker_df["el"]=="C"]["dist2"].min()].index.tolist()[0]

        linkerAnchorVec = linkerAnchor1POS - linkerAnchor2POS

        rotmat = rotmat2align(-linkerAnchorVec, anchorPairVec)
        linker_df.loc[:, ["x", "y", "z"]] = rotmat.dot(linker_df.loc[:, ["x", "y", "z"]].values.T).T

        displacementVec = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values - \
                          linker_df.loc[linkerCarbon1Idx, ["x", "y", "z"]].astype(float).values
        linker_df.loc[:, ["x", "y", "z"]] = linker_df.loc[:, ["x", "y", "z"]].values + displacementVec

        ldf.append(linker_df[linker_df["el"]!=dummyElement].copy(deep=True))
        farAnchorNLPairDispVec = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values - \
                                 linker_df.loc[linkerCarbon2Idx, ["x", "y", "z"]].astype(float).values
        latVec.append(farAnchorNLPairDispVec)

    final_df = pd.concat(ldf + [node_df[node_df["el"]!=dummyElement].copy(deep=True)], axis=0)
    final_df = final_df[final_df["el"]!=dummyElement].reset_index(drop=True)[["el", "x", "y", "z"]]
    pandas2xyzfile(final_df, os.path.join(newMOFdir, "mofCart.xyz"))

    mol = mg.Molecule.from_file(os.path.join(newMOFdir, "mofCart.xyz"))
    mg.Structure(np.array(latVec), 
                 mol.species, 
                 mol.cart_coords, 
                 coords_are_cartesian=True).to(filename=os.path.join(newMOFdir, "mofCart.cif"), 
                                               fmt="cif")
    return os.path.join(newMOFdir, "mofCart.cif")

def assemble_pillaredPaddleWheel_pcuMOF(nodePath, 
                                        LinkerPaths, 
                                        PillarLinkerPath, 
                                        newMOFdir, 
                                        dummyElementCOO="At", 
                                        dummyElementPillar="Fr"):
    os.makedirs(newMOFdir, exist_ok=True)

    node_df, nodeAnchorsCOO, nodeAnchorsPillar = readPillaredPaddleWheelXYZ(nodePath)

    linkerAnchors = [None for x in range(2)]
    linker_dfs = [None for x in range(2)]
    linker_dfs[0], linkerAnchors[0] = readLinkerXYZ(LinkerPaths[0])
    linker_dfs[1], linkerAnchors[1] = readLinkerXYZ(LinkerPaths[1])
    Pillarlinker_df, PillarlinkerAnchors = readNbasedLinkerXYZ(PillarLinkerPath)
    ldf = []
    latVec = []

    noderAnchorPair = nodeAnchorsPillar
    nodeAnchor1POS = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values
    nodePaired2POS = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values
    anchorPairVec = nodeAnchor1POS - nodePaired2POS
    linker_df = Pillarlinker_df
    linkerAnchorPair = PillarlinkerAnchors

    # translate to align n1 and l1
    linkerAnchor1POS = linker_df.loc[linkerAnchorPair[0], ["x", "y", "z"]].astype(float).values
    linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS) * \
                                (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS), 
                                axis=1)
    linkerNitrogen1Idx = linker_df[linker_df["dist2"]==linker_df[linker_df["el"]=="N"]["dist2"].min()].index.tolist()[0]

    linkerAnchor2POS = linker_df.loc[linkerAnchorPair[1], ["x", "y", "z"]].astype(float).values
    linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS) * \
                                (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS), 
                                axis=1)
    linkerNitrogen2Idx = linker_df[linker_df["dist2"]==linker_df[linker_df["el"]=="N"]["dist2"].min()].index.tolist()[0]

    linkerAnchorVec = linkerAnchor1POS - linkerAnchor2POS

    rotmat = rotmat2align(-linkerAnchorVec, anchorPairVec)
    linker_df.loc[:, ["x", "y", "z"]] = rotmat.dot(linker_df.loc[:, ["x", "y", "z"]].values.T).T

    displacementVec = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values - \
                      linker_df.loc[linkerNitrogen1Idx, ["x", "y", "z"]].astype(float).values
    linker_df.loc[:, ["x", "y", "z"]] = linker_df.loc[:, ["x", "y", "z"]].values + displacementVec

    ldf.append(linker_df[linker_df["el"]!=dummyElementPillar].copy(deep=True))
    farAnchorNLPairDispVec = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values - \
                             linker_df.loc[linkerNitrogen2Idx, ["x", "y", "z"]].astype(float).values
    latVec.append(farAnchorNLPairDispVec)

    for _i_ in range(len(nodeAnchorsCOO)):
        noderAnchorPair = nodeAnchorsCOO[_i_]
        nodeAnchor1POS = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values
        nodePaired2POS = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values
        anchorPairVec = nodeAnchor1POS - nodePaired2POS
        linker_df = linker_dfs[_i_]
        linkerAnchorPair = linkerAnchors[_i_]

        # translate to align n1 and l1
        linkerAnchor1POS = linker_df.loc[linkerAnchorPair[0], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS) * \
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor1POS), 
                                    axis=1)
        linkerCarbon1Idx = linker_df[linker_df["dist2"]==linker_df[linker_df["el"]=="C"]["dist2"].min()].index.tolist()[0]

        linkerAnchor2POS = linker_df.loc[linkerAnchorPair[1], ["x", "y", "z"]].astype(float).values
        linker_df["dist2"] = np.sum((linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS) * \
                                    (linker_df.loc[:, ["x", "y", "z"]].astype(float).values - linkerAnchor2POS), 
                                    axis=1)
        linkerCarbon2Idx = linker_df[linker_df["dist2"]==linker_df[linker_df["el"]=="C"]["dist2"].min()].index.tolist()[0]

        linkerAnchorVec = linkerAnchor1POS - linkerAnchor2POS

        rotmat = rotmat2align(-linkerAnchorVec, anchorPairVec)
        linker_df.loc[:, ["x", "y", "z"]] = rotmat.dot(linker_df.loc[:, ["x", "y", "z"]].values.T).T

        displacementVec = node_df.loc[noderAnchorPair[0], ["x", "y", "z"]].astype(float).values - \
                          linker_df.loc[linkerCarbon1Idx, ["x", "y", "z"]].astype(float).values
        linker_df.loc[:, ["x", "y", "z"]] = linker_df.loc[:, ["x", "y", "z"]].values + displacementVec

        ldf.append(linker_df[linker_df["el"]!=dummyElementCOO].copy(deep=True))
        farAnchorNLPairDispVec = node_df.loc[noderAnchorPair[1], ["x", "y", "z"]].astype(float).values - \
                                 linker_df.loc[linkerCarbon2Idx, ["x", "y", "z"]].astype(float).values
        latVec.append(farAnchorNLPairDispVec)


    final_df = pd.concat(ldf + [node_df[(node_df["el"]!=dummyElementCOO)&(node_df["el"]!=dummyElementPillar)].copy(deep=True)], axis=0)
    final_df = final_df[final_df["el"]!=dummyElementCOO].reset_index(drop=True)[["el", "x", "y", "z"]]
    pandas2xyzfile(final_df, os.path.join(newMOFdir, "mofCart.xyz"))

    mol = mg.Molecule.from_file(os.path.join(newMOFdir, "mofCart.xyz"))
    mg.Structure(np.array(latVec), 
                 mol.species, 
                 mol.cart_coords, 
                 coords_are_cartesian=True).to(filename=os.path.join(newMOFdir, "mofCart.cif"), 
                                               fmt="cif")
    return os.path.join(newMOFdir, "mofCart.cif")

def assemble_PillarPaddle_pcuMOF_multiProc(inputDict):
    nodePath = inputDict["nodePath"]
    linkerPaths = inputDict["linkerPaths"]
    PillarLinkerPath = inputDict["PillarLinkerPath"]
    newMOFdir = inputDict["newMOFdir"]
    if "dummyElementCOO" in inputDict.keys():
        dummyElementCOO = inputDict["dummyElementCOO"]
    else:
        dummyElementCOO = "At"

    if "dummyElementPillar" in inputDict.keys():
        dummyElementPillar = inputDict["dummyElementPillar"]
    else:
        dummyElementPillar = "Fr"

    returnValue = None
    try:
        returnValue = assemble_pillaredPaddleWheel_pcuMOF(nodePath, 
                                        linkerPaths, 
                                        PillarLinkerPath, 
                                        newMOFdir, 
                                        dummyElementCOO=dummyElementCOO, 
                                        dummyElementPillar=dummyElementPillar)
    except Exception as e: 
        print(e)
    return returnValue


if __name__ == "__main__":
    import os
    import multiprocessing as mproc
    NCPUS = int(0.9*os.cpu_count())

    newMOFdir = "newMOFs"
    os.makedirs(newMOFdir, exist_ok=True)
    nodes = ['ZnOZnZnZn','CuCu','ZnZn']
    for node in nodes:
        print(f'now on node: {node}')
        linkerPaths = sorted([os.path.join('linker_xyz',node,i) for i in os.listdir(os.path.join('linker_xyz',node))])
        linkerPaths_heterocyclic = sorted([os.path.join('linker_heterocyclic_xyz',node,i) for i in os.listdir(os.path.join('linker_heterocyclic_xyz',node))])
        nodePath = f"node_xyz/{node}.xyz"

        print("Running on " + str(NCPUS) + " processors...")

        if node == 'ZnOZnZnZn':
            l1_l2_l3_list = []
            # randomly select a given number of structures from all combinations
            for i in range(10000):
                l1_l2_l3_list.append(random_combination(linkerPaths,3))

            inputDicts = []
            for i in tqdm(range(len(l1_l2_l3_list))):
                x = l1_l2_l3_list[i][0]
                y = l1_l2_l3_list[i][1]
                z = l1_l2_l3_list[i][2]

                inputDicts.append(
                {"linkerPaths": [x, y, z],
                "nodePath": nodePath,
                "newMOFdir": os.path.join(newMOFdir, 
                                        f"N-{node}-L-" + x.split("/")[-1].split(".xyz")[0]\
                                        + "-L-" + y.split("/")[-1].split(".xyz")[0]\
                                        + "-L-" + z.split("/")[-1].split(".xyz")[0])})
            with mproc.Pool(NCPUS) as mp:
                cifNames = mp.map_async(assemble_Zn_Tetramer_pcuMOF_multiProc, inputDicts).get()

        if node in ['CuCu','ZnZn']:
            l1_l2_list = []
            l3_list = []
            l1_l2_l3_list = []
            # randomly select a given number of structures from all combinations
            for i in range(10000):
                l1_l2_list.append(random_combination(linkerPaths,2))
            for i in range(10000):
                l3_list.append(random_combination(linkerPaths_heterocyclic,1))

            l1_l2_l3_list = [l1_l2_list[i]+l3_list[i] for i in range(len(l1_l2_list))]

            inputDicts = []
            for i in tqdm(range(len(l1_l2_l3_list))):
                x = l1_l2_l3_list[i][0]
                y = l1_l2_l3_list[i][1]
                z = l1_l2_l3_list[i][2]

                inputDicts.append(
                {"linkerPaths": [x, y],
                "PillarLinkerPath": z,
                "nodePath": nodePath,
                "newMOFdir": os.path.join(newMOFdir, 
                                        f"N-{node}-L-" + x.split("/")[-1].split(".xyz")[0]\
                                        + "-L-" + y.split("/")[-1].split(".xyz")[0]\
                                        + "-L-" + z.split("/")[-1].split(".xyz")[0])})
            with mproc.Pool(NCPUS) as mp:
                cifNames = mp.map_async(assemble_PillarPaddle_pcuMOF_multiProc, inputDicts).get()