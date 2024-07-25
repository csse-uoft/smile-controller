#from smile_controller.listener import Text, Trace, Ks, KSAR, Hypothesis, Text, Ks, KSAR, Sentence, Outcome, Program, Service, BeneficialStakeholder, Organization, Phrase


import re, os, tqdm, json
from owlready2 import default_world, onto_path, ObjectProperty, DataProperty, rdfs, Thing 
onto_path.append('./ontology_cache/')
from py2graphdb.config import config as CONFIG
smile = default_world.get_ontology(CONFIG.NM)
with smile:
    from py2graphdb.Models.graph_node import GraphNode, SPARQLDict
    from py2graphdb.utils.db_utils import resolve_nm_for_ttl, resolve_nm_for_dict, PropertyList

    from smile_base.Model.knowledge_source.knowledge_source import KnowledgeSource
    from smile_base.Model.data_level.cids.organization import Organization
    from smile_base.Model.data_level.cids.program import Program
    from smile_base.Model.data_level.cids.service import Service
    from smile_base.Model.data_level.cids.stakeholder import Stakeholder
    from smile_base.Model.data_level.cids.beneficial_stakeholder import BeneficialStakeholder
    from smile_base.Model.data_level.cids.outcome import Outcome
    # from smile_base.Model.data_level.CatchmentArea"         : 'catchment_area',
    from smile_base.Model.data_level.query import Query
    from smile_base.Model.data_level.phrase import Phrase
    from smile_base.Model.data_level.text import Text
    from smile_base.Model.data_level.sentence import Sentence
    from smile_base.Model.data_level.hypothesis import Hypothesis
    from smile_base.Model.controller.ks import Ks
    from smile_base.Model.controller.ks_ar import KSAR
    from smile_base.Model.controller.trace import Trace

from py2graphdb.ontology.operators import *