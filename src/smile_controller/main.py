from owlready2 import default_world,onto_path, ObjectProperty, DataProperty, rdfs, Thing 
onto_path.append('./smile_controller/ontology_cache/')
import re, os, tqdm, pandas as pd
# from smile_controller.listener import ParseQuery, Query, Text, Trace, Ks, KSAR, Hypothesis, Text, Ks, KSAR, Phrase, Sentence

from smile_controller.listener import Query, Text, Trace, Ks, KSAR, Hypothesis, Text, Ks, KSAR, Sentence, Outcome, Program, Service, BeneficialStakeholder, Organization, Phrase
from py2graphdb.config import config as CONFIG
from py2graphdb.Models.graph_node import GraphNode, SPARQLDict
from py2graphdb.utils.db_utils import resolve_nm_for_dict, PropertyList, _resolve_nm
from py2graphdb.ontology.namespaces import ic, geo, cids, org, time, schema, sch, activity, landuse_50872, owl
from py2graphdb.ontology.operators import *
if not os.path.exists(CONFIG.LOG_DIR):
    os.makedirs(CONFIG.LOG_DIR)
smile = default_world.get_ontology(CONFIG.NM)
from smile_base.utils import init_db
from smile_controller.libs.schedule import Schedule, Task





if not os.path.exists(CONFIG.LOG_DIR):
    os.makedirs(CONFIG.LOG_DIR)


description = "St.Mary's Church provides hot meals &amp; addiction support to 90% of homeless youth to support a brighter future."
#description = "About Oak Trees and Acorns. A Fun and Enriched Experience. At Oak Trees and Acorns we believe in providing children with a supportive environment which emphasizes quality of life, encourages growth, and recognizes the importance of family and of generations coming together. We believe in order for children to learn, they must first feel good about themselves and feel confident in their capabilities. We strive to provide a variety of experiences that are age appropriate to a child's learning style and developmental needs. Children enrolled at our Sherbrooke (long term care home) location will become an integral part of daily life at Sherbrooke. We believe that intergenerational interaction between children and residents strengthens the whole community, helps encourage the child's self worth through an extended family concept, and improves the quality of life for all. Children enrolled at our Caroline Robins location will become an integral part of the community school environment. We believe that early exposure to the school community, its activities and resources, create more seamless transitions to Pre Kindergarten and Kindergarten education, while helping to create important social bonds within the community. Children at this location will also have the same opportunity for intergenerational interactions with Central Haven, Sherbrooke's sister site, although on a less frequent spontaneous schedule. Children enrolled in both locations will be enriched with our inclusive programming using the Play and Exploration curriculum. We are licensed through the Ministry of Education, Early Years Branch and regulated by The Child Care Act, and The Saskatchewan Child Care Regulations. www.education.gov.sk.ca/elcc. Are you ready to start your future with us?"
# description = "At Oak Trees and Acorns we believe in providing children with a supportive environment which emphasizes quality of life, encourages growth, and recognizes the importance of family and of generations coming together. We believe in order for children to learn, they must first feel good about themselves and feel confident in their capabilities. We strive to provide a variety of experiences that are age appropriate to a child's learning style and developmental needs. Children enrolled at our Sherbrooke (long term care home) location will become an integral part of daily life at Sherbrooke. We believe that intergenerational interaction between children and residents strengthens the whole community, helps encourage the child's self worth through an extended family concept, and improves the quality of life for all. Children enrolled at our Caroline Robins location will become an integral part of the community school environment."


def reset():
    with smile:
        init_db.init_db()
        init_db.load_owl('./tmp/init-ontology.owl')
        init_db.load_owl('./smile_controller/ontology_cache/cids.ttl')

        Schedule.set_kss()
def dfs():
    with smile:

        import sys
        trace = Trace(keep_db_in_synch=True)
        print("Trace", trace)
        hypothesis = Query.find_generate(content=description, trace_id=trace.id)
        hypothesis.save()
        schedule = Schedule(trace=trace)
        schedule.cycle=0
        res = schedule.subschedule_run(hypothesis=hypothesis, search='dfs', trigger_event=f"Received new {type(hypothesis)}")
        return trace

def show_trace(trace_id):
    def show_phrases(nodes):
        ranked_nodes = sorted(nodes, key=lambda x: round(x.certainty,3), reverse=True)

        for out in ranked_nodes:
            try:
                print(out, "\t", out.trace[:20], round(out.certainty, 3))
                print("\t\t", Phrase(out.phrase).content)
            except:
                pass    
    def save_phrases(nodes):
        columns = ['trace', 'klass', 'id', 'certainty', 'phrase_id', 'content','phrase_certainty']
        df = pd.DataFrame(columns=['trace', 'klass', 'id', 'certainty', 'phrase_id', 'content','phrase_certainty'])
        ranked_nodes = sorted(nodes, key=lambda x: round(x.certainty, 3), reverse=True)

        for out in ranked_nodes:
            try:
                phrase = Phrase(out.phrase)
                df = pd.concat([df, pd.DataFrame([[out.trace, out.klass, out.id, round(out.certainty, 3), phrase.id, phrase.content, round(phrase.certainty, 3)]],columns=columns)])
            except:
                pass    
            if df.empty:
                df = pd.DataFrame(columns=['trace', 'klass', 'id', 'certainty', 'phrase_id', 'content','phrase_certainty'])
        return df

    trace = Trace(trace_id)
    programs = Program.search(props={smile.hasTraceID:trace.id}, how='all')
    outcomes = Outcome.search(props={smile.hasTraceID:trace.id}, how='all')
    bss = BeneficialStakeholder.search(props={smile.hasTraceID:trace.id}, how='all')
    services = Service.search(props={smile.hasTraceID:trace.id}, how='all')
    orgs = Organization.search(props={smile.hasTraceID:trace.id}, how='all')

    print('Programs')
    show_phrases(programs)
    print('Service')
    show_phrases(services)
    print('Outcomes')
    show_phrases(outcomes)
    print('BeneficialStakeholders')
    show_phrases(bss)
    print('Organization')
    show_phrases(orgs)

    df = save_phrases(programs)
    df = pd.concat([df, save_phrases(services)])
    df = pd.concat([df, save_phrases(outcomes)])
    df = pd.concat([df, save_phrases(bss)])
    df = pd.concat([df, save_phrases(orgs)])
    df.to_csv(f'{CONFIG.LOG_DIR}concepts-{trace_id}.csv', index=False)

##################################################################################
def improve_bfs(hypothesis, trigger_event):
    print('\timprove_bfs()')
    print('\t1', type(hypothesis), hypothesis)
    print('\t2', type(hypothesis.trace), hypothesis.trace)
    trace = Trace.get(hypothesis.trace)
    print('\t3')
    schedule = Schedule(trace=trace)
    print('\t4')
    # schedule.cycle=0
    print('\t5')
    _ = schedule.subschedule_improve(search='bfs', hypothesis=hypothesis, trigger_event=trigger_event)
    print('\t6')
    return schedule

def process0_bfs(hypothesis, trigger_event):
    trace = Trace.get(hypothesis.trace)
    schedule = Schedule(trace=trace)
    schedule.incr_cycle()
    _ = schedule.subschedule_process0(search='bfs', hypothesis=hypothesis, trigger_event=trigger_event)
    return schedule
def process1_bfs(hypothesis, trigger_event):
    trace = Trace.get(hypothesis.trace)
    schedule = Schedule(trace=trace)
    schedule.incr_cycle()
    _ = schedule.subschedule_process1(search='bfs', hypothesis=hypothesis, trigger_event=trigger_event)
    return schedule

##################################################################################
def improve_bfs_loop(top_hypothesis, trigger_event):
    prev_top_certainty = round(top_hypothesis.certainty, 3)
    while True:
        schedule = improve_bfs(top_hypothesis, trigger_event=trigger_event)
        ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty, 3), reverse=True)
        if len(ranked_outputs)>0:
            new_top_hypothesis = ranked_outputs[0]
        else:
            break
        if top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) == prev_top_certainty:
            # no global improvements
            trigger_event = "No new local improvements on top hypothesis"
            break
        elif top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) != prev_top_certainty:
            trigger_event = "Top local hypothesis certainty updated"
        elif top_hypothesis.id != new_top_hypothesis.id:
            trigger_event = "New top local hypothesis found"
        top_hypothesis = new_top_hypothesis
        prev_top_certainty = round(top_hypothesis.certainty, 3)
    return top_hypothesis

def main_top_global_bfs(top_hypothesis, trigger_event):
    trace = Trace(top_hypothesis.trace)
    with smile:
        prev_top_certainty = round(top_hypothesis.certainty, 3)
        new_top_hypothesis = None
        prev_hypothesis = top_hypothesis
        while True:
            print("restart with ", top_hypothesis)
            tmp_top_hypothesis = improve_bfs_loop(top_hypothesis=top_hypothesis, trigger_event=trigger_event)
            if new_top_hypothesis is not None and tmp_top_hypothesis is not None and tmp_top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) == prev_top_certainty:
                trigger_event = "Top hypothesis cannot be improved. Finding new global top hypothesis that can be processed."
                top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='find global higher Hypothesis 1',trace_id=trace.id, subclass=True)
                if top_hypothesis is None or len(possible_kss) == 0:
                    trigger_event = "No Hypotheses to processes"
                    print("\t>>te0", trigger_event)
                    break
                else:
                    trigger_event = "Found new global highest Hypothesis. Improving KSs"
                    top_hypothesis = improve_bfs_loop(top_hypothesis=top_hypothesis, trigger_event=trigger_event)
            else:
                trigger_event='Processing top local hypothesis with no other inputs.'
            schedule = process0_bfs(top_hypothesis, trigger_event=trigger_event)
            ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty,3), reverse=True)
            if len(ranked_outputs)>0:
                new_top_hypothesis = ranked_outputs[0]
            else:
                trigger_event = "Processing top local hypothesis with extra inputs."
                schedule = process1_bfs(top_hypothesis, trigger_event=trigger_event)
                ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty,3), reverse=True)
                if len(ranked_outputs)>0:
                    new_top_hypothesis = ranked_outputs[0]
                else:
                    trigger_event = "No new hypotheses found. Finding new global top hypothesis that can be processed."
                    top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='find global higher Hypothesis 2', trace_id=trace.id, subclass=True)
                    if top_hypothesis is None or len(possible_kss) == 0:
                        trigger_event = "No Hypotheses to processes"
                        print("\t>>te1", trigger_event)
                        break
                    else:
                        trigger_event = "Found new global highest hypothesis."
            print("\tcheck1:",  type(prev_hypothesis), type(top_hypothesis), type(new_top_hypothesis), type(top_hypothesis), type(prev_top_certainty))
            print("\tcheck2:",  prev_hypothesis, top_hypothesis, new_top_hypothesis, top_hypothesis, prev_top_certainty)
            #print("\tcheck3:",  prev_hypothesis.id, top_hypothesis.id, new_top_hypothesis.id, round(top_hypothesis.certainty, 3), prev_top_certainty)
            if new_top_hypothesis is not None:
                if top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) == prev_top_certainty:
                    # no global improvements
                    trigger_event = "No new local improvements on top hypothesis"
                    print("\t>>te2", trigger_event)
                    break
                elif top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) != prev_top_certainty:
                    trigger_event = "Top local hypothesis certainty updated"
                elif prev_hypothesis is not None and prev_hypothesis.id != new_top_hypothesis.id:
                    trigger_event = "New top local hypothesis found 1"
                elif top_hypothesis.id != new_top_hypothesis.id:
                    trigger_event = "New top local hypothesis found 2"
                print("\tEND te", trigger_event)
                top_hypothesis = new_top_hypothesis
                prev_top_certainty = round(top_hypothesis.certainty, 3)
                prev_hypothesis = top_hypothesis
                new_top_hypothesis = top_hypothesis


def main_top_global_bfs_init():
    with smile:
        trace = Trace(keep_db_in_synch=True)
        print("Trace", trace)
        hypothesis = Query.find_generate(content=description, trace_id=trace.id)
        hypothesis.save()
        trigger_event='Got new query; improving new hypothesis'
        main_top_global_bfs(top_hypothesis=hypothesis, trigger_event=trigger_event)
        top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='Find new global highest Hypothesis', trace_id=trace.id, subclass=True)
        while top_hypothesis is not None:
            trigger_event = 'Restarted search and found new global highest Hypothesis'
            main_top_global_bfs(top_hypothesis=hypothesis, trigger_event=trigger_event)
            top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='Find new global highest Hypothesis', trace_id=trace.id, subclass=True)

        show_trace(trace.inst_id)

######################################################################
            
def improve_bfs_loop(top_hypothesis, trigger_event):
    prev_top_certainty = round(top_hypothesis.certainty, 3)
    while True:
        schedule = improve_bfs(top_hypothesis, trigger_event=trigger_event)
        ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty,3), reverse=True)
        if len(ranked_outputs)>0:
            new_top_hypothesis = ranked_outputs[0]
        else:
            break
        if top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) == prev_top_certainty:
            # no global improvements
            trigger_event = "No new local improvements on top hypothesis"
            break
        elif top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty, 3) != prev_top_certainty:
            trigger_event = "Top local hypothesis certainty updated"
        elif top_hypothesis.id != new_top_hypothesis.id:
            trigger_event = "New top local hypothesis found"
        top_hypothesis = new_top_hypothesis
        prev_top_certainty = round(top_hypothesis.certainty, 3)
    return top_hypothesis

def main_top_local_bfs(top_hypothesis, trigger_event):
    trace = Trace(top_hypothesis.trace)
    with smile:
        prev_top_certainty = round(top_hypothesis.certainty, 3)
        new_top_hypothesis = None
        prev_hypothesis = top_hypothesis
        while True:
            print("restart with ", top_hypothesis)
            tmp_top_hypothesis = improve_bfs_loop(top_hypothesis=top_hypothesis, trigger_event=trigger_event)
            if new_top_hypothesis is not None and tmp_top_hypothesis is not None and tmp_top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) == prev_top_certainty:
                trigger_event = "Top hypothesis cannot be improved. Finding new global top hypothesis that can be processed."
                ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty,3), reverse=True)

                top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='find global higher Hypothesis 1',trace_id=trace.id, subclass=True)
                if top_hypothesis is None or len(possible_kss) == 0:
                    trigger_event = "No Hypotheses to processes"
                    print("\t>>te0", trigger_event)
                    break
                else:
                    trigger_event = "Found new global highest Hypothesis. Improving KSs"
                    top_hypothesis = improve_bfs_loop(top_hypothesis=top_hypothesis, trigger_event=trigger_event)
            else:
                trigger_event='Processing top local hypothesis with no other inputs.'
            schedule = process0_bfs(top_hypothesis, trigger_event=trigger_event)
            ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty,3), reverse=True)
            if len(ranked_outputs)>0:
                new_top_hypothesis = ranked_outputs[0]
            else:
                trigger_event = "Processing top local hypothesis with extra inputs."
                schedule = process1_bfs(top_hypothesis, trigger_event=trigger_event)
                ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty,3), reverse=True)
                if len(ranked_outputs)>0:
                    new_top_hypothesis = ranked_outputs[0]
                else:
                    trigger_event = "No new hypotheses found. Finding new global top hypothesis that can be processed."
                    top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='find global higher Hypothesis 2', trace_id=trace.id, subclass=True)
                    if top_hypothesis is None or len(possible_kss) == 0:
                        trigger_event = "No Hypotheses to processes"
                        print("\t>>te1", trigger_event)
                        break
                    else:
                        trigger_event = "Found new global highest hypothesis."
            print("\tcheck:",  prev_hypothesis.id, top_hypothesis.id, new_top_hypothesis.id, round(top_hypothesis.certainty,3), prev_top_certainty)
            if top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) == prev_top_certainty:
                # no global improvements
                trigger_event = "No new local improvements on top hypothesis"
                print("\t>>te2", trigger_event)
                break
            elif top_hypothesis.id == new_top_hypothesis.id and round(top_hypothesis.certainty,3) != prev_top_certainty:
                trigger_event = "Top local hypothesis certainty updated"
            elif prev_hypothesis is not None and prev_hypothesis.id != new_top_hypothesis.id:
                trigger_event = "New top local hypothesis found 1"
            elif top_hypothesis.id != new_top_hypothesis.id:
                trigger_event = "New top local hypothesis found 2"
            print("\tEND te", trigger_event)
            top_hypothesis = new_top_hypothesis
            prev_top_certainty = round(top_hypothesis.certainty,3)
            prev_hypothesis = top_hypothesis
            new_top_hypothesis = top_hypothesis


def main_top_local_bfs_init():
    with smile:
        trace = Trace(keep_db_in_synch=True)
        print("Trace", trace)
        hypothesis = Query.find_generate(content=description, trace_id=trace.id)
        hypothesis.save()
        trigger_event='Got new query; improving new hypothesis'
        main_top_local_bfs(top_hypothesis=hypothesis, trigger_event=trigger_event)
        top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='Find new global highest Hypothesis', trace_id=trace.id, subclass=True)
        while top_hypothesis is not None:
            trigger_event = 'Restarted search and found new global highest Hypothesis'
            main_top_global_bfs(top_hypothesis=hypothesis, trigger_event=trigger_event)
            top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='Find new global highest Hypothesis', trace_id=trace.id, subclass=True)

        show_trace(trace.inst_id)

def main_top_local_init():
    with smile:
        trace = Trace(keep_db_in_synch=True)
        print("Trace", trace)
        hypothesis = Query.find_generate(content=description, trace_id=trace.id)
        hypothesis.save()
        trigger_event='Got new query; improving new hypothesis'
        # main_top_local_bfs(top_hypothesis=hypothesis, trigger_event=trigger_event)
        top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='Find new global highest Hypothesis', trace_id=trace.id, subclass=True)
        while top_hypothesis is not None:
            trigger_event = 'Restarted search and found new global highest Hypothesis'
            schedule = process0_bfs(top_hypothesis, trigger_event=trigger_event)
            # ranked_outputs = sorted(schedule.collect_outputs(), key=lambda x: round(x.certainty,3), reverse=True)
            main_top_global_bfs(top_hypothesis=hypothesis, trigger_event=trigger_event)
            top_hypothesis, possible_kss = Schedule.get_highest_ranked_global_hypo(label='Find new global highest Hypothesis', trace_id=trace.id, subclass=True)

            show_trace(trace.inst_id)



if __name__ == '__main__':
<<<<<<< HEAD
    main_top_global_bfs_init()
    # pass
    # main_top_local_bfs_init()
    # main_top_local_init()
    # trace = dfs()
    # show_trace(trace.inst_id)
=======
    main_top_local_bfs_init()
>>>>>>> 552c7b87f4c9905421194eb16e5f0bdc7e8c8a15
    
