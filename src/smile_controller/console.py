from owlready2 import default_world,onto_path, ObjectProperty, DataProperty, rdfs, Thing 
onto_path.append('./smile_controller/ontology_cache/')
import re, os, tqdm
from smile_controller.listener import Query, Text, Trace, Ks, KSAR, Hypothesis, Text, Ks, KSAR, Sentence, Outcome, Program, BeneficialStakeholder, Service

from py2graphdb.config import config as CONFIG
from py2graphdb.utils.db_utils import resolve_nm_for_dict, PropertyList, _resolve_nm
from py2graphdb.ontology.namespaces import ic, geo, cids, org, time, schema, sch, activity, landuse_50872, owl
from py2graphdb.ontology.operators import *
if not os.path.exists(CONFIG.LOG_DIR):
    os.makedirs(CONFIG.LOG_DIR)
smile = default_world.get_ontology(CONFIG.NM)
from smile_base.utils import init_db
from smile_controller.libs.schedule import Schedule, Task


# with smile:
#     init_db.init_db()
#     init_db.load_owl('./tmp/init-ontology.owl')
#     init_db.load_owl('./smile_controller/ontology_cache/cids.ttl')
