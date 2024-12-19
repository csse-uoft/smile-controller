from owlready2 import default_world,onto_path, ObjectProperty, DataProperty, rdfs, Thing 
onto_path.append('./smile_controller/ontology_cache/')
import re, os, tqdm, numpy as np, itertools as itert
from os.path import expanduser


import sys

from smile_controller.listener import Query, Text, OrgCertainty, Trace, Ks, KSAR, Hypothesis, Text, Ks, KSAR, Sentence, Outcome, Program, BeneficialStakeholder, Service, Phrase

from py2graphdb.config import config as CONFIG
from py2graphdb.utils.db_utils import resolve_nm_for_dict, PropertyList, _resolve_nm
from py2graphdb.ontology.namespaces import ic, geo, cids, org, time, schema, sch, activity, landuse_50872, owl
from py2graphdb.ontology.operators import *
if not os.path.exists(CONFIG.LOG_DIR):
    os.makedirs(CONFIG.LOG_DIR)
smile = default_world.get_ontology(CONFIG.NM)
from smile_base.utils import init_db
from smile_controller.libs.schedule import Schedule, Task

import networkx as nx


TRACE_ID = sys.argv[1]
DIR = sys.argv[2] if len(sys.argv)>1 else '.'
if DIR.startswith('~'):
    DIR = expanduser('~') + DIR[1:]
DIR += f"/{TRACE_ID}"
if not os.path.exists(DIR):
    os.makedirs(DIR)


# with smile:
#     init_db.init_db()
#     init_db.load_owl('./tmp/init-ontology.owl')
#     init_db.load_owl('./smile_controller/ontology_cache/cids.ttl')

trunc = lambda x,l=4: ''.join([xx+ (' ' if (i+1)%l!=0 else "\n") for i,xx in enumerate(x.split(' '))])

from matplotlib import pyplot as plt

def draw_me(g, pos, title="Run: all cycles", edge_color=None, figsize=None, path=None):
    plt.close()
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    if edge_color is None:
        edge_color = [g[u][v]['color'] for u,v in g.edges()]
    nx.draw(
        g, pos, edge_color=edge_color, width=1, linewidths=0.1,
        node_size=3000, node_color='yellow', alpha=0.7, font_size=8,
        labels={node: node for node in g.nodes()},
        arrowsize=20#, arrowstyle='fancy'
    )    
    edge_labels = dict([((n1, n2), f"{d.get('title')}")       for n1, n2,d in g.edges(data=True)])
    connectionstyle = "arc3,rad=0.1" #','.join([f"arc3,rad={r}" for r in itert.accumulate([0.15] * (len(edge_labels)+1))])
    # connectionstyle = str([f"arc3,rad={r}" for r in itert.accumulate([0.15] * 4)])
    # nx.draw_networkx_edges(
    #     g, 
    #     pos, 
    #     edgelist=edge_labels,
    #     edge_color="grey", 
    #     connectionstyle=connectionstyle
    # )
    nx.draw_networkx_edge_labels(
        g, pos,
        # connectionstyle=connectionstyle,
        edge_labels=edge_labels,
        font_color='black', font_size=10,
        label_pos=np.random.random()/3+0.3,
        # bbox={"alpha": 0}
    )
    plt.axis('off')
    plt.tight_layout()
    plt.title(title)
    if path is None:
        plt.show()
    else:
        fig.savefig(path, bbox_inches='tight')


klass_prefixes = {
    'Organization': 'Org',
    'Program': "P",
    'Service': "S",
    "BeneficialStakeholder": "B",
    'Outcome': 'Out',
    'Text': 'Text',
    'Sentence': 'Sent'
}
ks_colors = {
    'GPT': 'r',
    'QA': 'g'
}
def build_graph(edge_colors):
    hypos = [h for h in trace_hypos if isinstance(h,(smile.Program, smile.Service, smile.BeneficialStakeholder, smile.Outcome, smile.Organization))]
    texts = []
    sentences = []
    cycles = []
    g = nx.DiGraph()
    for hypo in tqdm.tqdm(hypos, total=len(hypos), desc='build graph'):
        content = Phrase(hypo.phrase).content
        # content = Phrase(hypo.phrase).content
        klass_node = klass_prefixes[re.sub(r'.*\.','',str(type(hypo)))]
        node = trunc(f"{klass_node}: {content}", l=3)
        _=g.add_node(node)
        _=g.add_edge(klass_node, node)
        ksars = [KSAR.get(ksar) for ksar in hypo.from_ks_ars]
        for ksar in ksars:
            inputs = ksar.input_hypotheses
            org_certainties = ksar.org_certainties
            # [h for h in kk.hypotheses if h==hypo.id][0]
            ks = Ks.get(ksar.ks).cast_to_graph_type()
            ks_node = ks.name.split(' ')[0].replace('ChatGPTPrompt', 'GPT')
            cycle_node = f"Cycle({ksar.cycle})"
            cycles.append(cycle_node)
            edge_colors = [vc for kc,vc in ks_colors.items() if ks_node.startswith(kc)]
            if edge_colors != '':
                edge_color = edge_colors[0]
            else:
                edge_color = 'lightgrey'
            for input_ in inputs:
                in_h = Hypothesis.get(input_).cast_to_graph_type()
                if isinstance(in_h, (smile.Text, smile.Sentence)):
                    if len(inputs)>1:
                        continue
                    # in_content = trunc(Phrase(in_h.phrase).content, l=3)
                    in_klass = klass_prefixes[re.sub(r'.*\.','',str(type(in_h)))]
                    in_content = trunc(f"{in_klass}: {in_h.content}",l=10)
                    if isinstance(in_h, smile.Text):
                        texts.append(in_content)
                    elif isinstance(in_h, smile.Sentence):
                        sentences.append(in_content)
                else:
                    try:
                        # in_content = trunc(Phrase(in_h.phrase).content, l=3)
                        in_klass = klass_prefixes[re.sub(r'.*\.','',str(type(in_h)))]
                        in_content = trunc(f"{in_klass}: {Phrase(in_h.phrase).content}", l=3)
                    except:
                        in_klass = re.sub(r'.*\.','',str(type(in_h)))
                        in_show = re.sub(r'.*\.','',str(in_h.show()))
                        in_content = trunc(f"{in_klass}: {in_show}", l=3)
                
                org_certs = list(set(hypo.org_certainties).intersection(org_certainties))
                if len(org_certs)==1:
                    org_cert = OrgCertainty.get(org_certs[0])
                else:
                    org_cert = OrgCertainty.get(hypo.org_certainties[0])
                edge_label = f"{round(org_cert.certainty,3)}\n(cycle={ksar.cycle})"
                _=g.add_edge(in_content, node, color=edge_color, title=edge_label)
                _=g.add_edge(cycle_node, in_content, color=edge_color, title='cycle')
                _=g.add_edge(cycle_node, node, color=edge_color, title='cycle')
    return g, texts, sentences, cycles

def offset(x, n=10):
    r = np.random.random()
    return n*(r  if r > 0.5 else -r) + x

def blackboard_layout(g, texts, sentences, klasses, sep=60):
    texts = list(set(texts))
    sentences = list(set(sentences))    
    klass_hypos = {}
    for klass in klasses:
        klass_hypos[klass] = list(g.neighbors(klass))
    sepx = sep
    sepy = sep
    x = 0
    y = -sep
    maxx = 0
    maxy = 0
    pos = {}
    for klass, k_hypos in klass_hypos.items():
        for i, h in enumerate(k_hypos):                
            pos[h] = np.array((offset(x,10),offset(y,10)))
            if i == 0:
                y = 0
            y += sepy
        maxx = max(maxx,x)
        maxy = max(maxy,y)
        y = -sep
        x += sepx
    # add sentences
    x = maxx/2.0
    y = -sep*2
    for h in sentences:
        pos[h] = np.array((offset(x,20),offset(y,10)))
        maxx = max(maxx,x)
        maxy = max(maxy,y)
        x += sepx
        y -= sepy
    # add texts
    y -= sepy
    x = maxx/2.0
    for h in texts:
        pos[h] = np.array((x,y))
        maxx = max(maxx,x)
        maxy = max(maxy,y)
        y += sepy
        x += sepx
    return pos

trace = Trace.get(TRACE_ID)
trace_hypos = [Hypothesis.get(h).cast_to_graph_type() for h in trace.hypotheses]

g, texts, sentences, cycles = build_graph(edge_colors=ks_colors)
cycles = list(set(cycles))
klasses = [klass_prefixes[k] for k in ['Organization', 'Program','Service','BeneficialStakeholder','Outcome']]
_=[g.add_node(k) for k in klasses]
g1 = g.copy()
g1.remove_nodes_from(cycles)
pos = blackboard_layout(g1, texts, sentences, klasses, sep=120)

def draw_full(g1, pos, klasses):
    xx,yy = np.array(list((x,y) for x,y in pos.values())).T
    size_x = int(np.max(xx) - np.min(xx))/30
    size_y = int(np.max(yy) - np.min(yy))/30
    g2 = g1.copy()
    g2.remove_nodes_from(klasses)
    draw_me(g2, pos=pos, figsize=(size_x, size_y), path=DIR+f"/cycle_all.pdf", title=f"Run: All cycles")

def draw_cycles(g, pos, cycles):
    xx,yy = np.array(list((x,y) for x,y in pos.values())).T
    size_x = int(np.max(xx) - np.min(xx))/30
    size_y = int(np.max(yy) - np.min(yy))/30
    cycle_idx = list(set([int(re.match(r'Cycle\((.+)\)', c)[1])for c in cycles]))
    cycle_idx.sort()
    for ci in cycle_idx:
        c = f"Cycle({ci})"
        g0 = g.subgraph(nx.neighbors(g, c))
        pos0 = dict((k, pos[k]) for k in g0.nodes())
        draw_me(g0, pos=pos0, figsize=(size_x, size_y), path=DIR+f"/cycle_{ci}.pdf", title=f"Run: {c}")


def draw_merged_cycles(g, pos, cycles):
    xx,yy = np.array(list((x,y) for x,y in pos.values())).T
    size_x = int(np.max(xx) - np.min(xx))/30
    size_y = int(np.max(yy) - np.min(yy))/30
    cycle_idx = list(set([int(re.match(r'Cycle\((.+)\)', c)[1])for c in cycles]))
    cycle_idx.sort()
    included_cycles = []
    for ci in tqdm.tqdm(cycle_idx, total=len(cycle_idx), desc='gen graphs'):
        c = f"Cycle({ci})"
        included_cycles.append(c)
        nodes = []
        for cc in included_cycles:
            nodes += list(nx.neighbors(g, cc)) 
        nodes = set(nodes)
        g0 = g.subgraph(nodes)
        pos0 = dict((k, pos[k]) for k in g0.nodes())
        draw_me(g0, pos=pos0, figsize=(size_x, size_y), path=DIR+f"/cycle_merged{ci}.pdf", title=f"Run: {included_cycles}")

draw_full(g1, pos, klasses)
draw_cycles(g, pos, cycles)
draw_merged_cycles(g, pos, cycles)

