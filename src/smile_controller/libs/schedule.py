import os, numpy as np
import types
from tqdm import tqdm
from collections import defaultdict

from py2graphdb.config import config as CONFIG
from py2graphdb.utils.db_utils import SPARQLDict
from __main__ import Ks, smile, Hypothesis, Trace, KSAR, Text
from smile_base.Model.knowledge_source.knowledge_source import KnowledgeSource
from py2graphdb.ontology.operators import *
from py2graphdb.Models.graph_node import GraphNode

import time
import itertools as itert

# todo:
#  search operations
# depth-first
# breadth-first
# island-search
# select by certainty (low/high)

class Task:
    _id = itert.count(0)
    def __init__(self, ks, local=False, inputs=[]) -> None:
        self.id = next(self._id)
        self.ks = ks
        self.name = ks.name
        self.py_name = ks.py_name
        self.ks_ar = None
        self.local = local
        self.inputs = inputs
        self.outputs = []
    def run(self):
        if self.ks_ar is not None:
            self.ks_ar.ks_status = 0
            if self.local:
                ks = Ks(self.ks_ar.ks)
                eval(f"{ks.py_name}.process_ks_ars(loop=False)")
            self.ks_ar.load()
            print(f'>>>>>{self.name}<<<<<', flush=True)
            print(f'   >>{self.ks_ar}<<<<<', flush=True)
            print(f'   >>{self.inputs}<<<<<', flush=True)
            print(f'   >>{[i.show() for i in self.inputs]}', flush=True)
            while self.ks_ar.ks_status is None or (self.ks_ar.ks_status >= 0 and self.ks_ar.ks_status < 3):
                print(self.ks_ar.ks_status, end='', flush=True)
                time.sleep(1)
                self.ks_ar.load()
            self.ks_ar.load()
            print('OUT', self.ks_ar.ks_status, flush=True)
        else:
            print('no ks_ar')

class Schedule:
    _last_cycle = 0
    _id = itert.count(0)
    def __init__(self, trace, inputs=[]) -> None:
        self.id = next(self._id)
        self.trace = trace
        self.inputs = inputs
        self.tasks = []
        self.outputs = []
        self.subschedules = []

    def incr_cycle(self):
        ksars = KSAR.search(props={smile.hasTraceID:self.trace.id}, how='all')
        if len(ksars)>0:
            last_cycle = sorted(ksars, key=lambda x: x.cycle, reverse=True)[0].cycle
        else:
            last_cycle = 0
        self.cycle = last_cycle + 1

    def collect_outputs(self):
        tmp_hypothesies = []
        for schedule in self.subschedules:
            tmp_hypothesies += schedule.collect_outputs()
        return tmp_hypothesies + self.outputs

    def gen_ksar(self, task, trigger_event=None):
        hypothesis = task.inputs[0]
        input_klasses = [i.klass for i in task.inputs]
        targets = list(set(task.ks.inputs).difference(input_klasses))
        input_klasses = [k for k in input_klasses if k != hypothesis.klass]
        matches = {}
        for ks_input in targets:
            print("\t\t## try 1", hypothesis, str(ks_input), task.ks.name, flush=True)
            try:
                res = []
                exists = SPARQLDict._process_path_request(start=hypothesis, end=str(ks_input), action='ask', direction='children', how='shortest', infer=False)
                if exists:
                    res = SPARQLDict._process_path_request(start=hypothesis, end=str(ks_input), action='collect', direction='children', how='shortest', infer=False)
            except ValueError:
                res = []
                # remove KSARs
                for r in res:
                    path = r['path']
                    for inst_id in path.copy():
                        node = GraphNode(inst_id).cast_to_graph_type()
                        if isinstance(node, (smile.Ks, KSAR)) and inst_id in path:
                            path.remove(inst_id)
                res = [r for r in res if len(r['path'])>0]
            if len(res)>0:
                for r in res:
                    # print(r)
                    if len(r['path'])>0:
                        matches[ks_input] = r['path'][-1] 
            else:
                print("\t\t## try 2", hypothesis, ks_input, task.ks.name, flush=True)
                try:
                    res = []
                    exists = SPARQLDict._process_path_request(end=hypothesis, start=str(ks_input), action='ask', direction='children', how='shortest', infer=False)
                    if exists:
                        res = SPARQLDict._process_path_request(end=hypothesis, start=str(ks_input), action='collect', direction='children', how='shortest', infer=False)
                except ValueError:
                    res = []
                    # remove KSARs
                    for r in res:
                        path = r['path']
                        for inst_id in path.copy():
                            node = GraphNode(inst_id).cast_to_graph_type()
                            if isinstance(node, (smile.Ks, KSAR)) and inst_id in path:
                                path.remove(inst_id)
                    res = [r for r in res if len(r['path'])>0]
                if len(res)>0:
                    for r in res:
                        # print(r)
                        if len(r['path'])>0:
                            matches[ks_input] = r['path'][0] 

        if len(set(targets).intersection(matches)) == len(targets):
            if len(matches)>0:
                tmp = [[hypothesis],list(matches.values())]
                input_hypos = list(itert.product(*tmp))
            else:
                input_hypos = [[hypothesis]]

            for inputs in input_hypos:
                ks_ar = KSAR()
                ks_ar.keep_db_in_synch = False
                ks_ar.ks = task.ks.id
                ks_ar.trace = self.trace.id
                ks_ar.cycle = self.cycle
                ks_ar.trigger_event = trigger_event or f"New task in {self.id}"
                for hypo in [hypothesis]+ [Hypothesis(id) for id in inputs]:
                    ks_ar.input_hypotheses = hypo.id
                    hypo.for_ks_ars = ks_ar.inst_id

                ks_ar.save()
                ks_ar.keep_db_in_synch = True
                task.ks_ar = ks_ar
                return ks_ar
        else:
            return None


    def run(self):
        for task in self.tasks:
            ks_ar = self.gen_ksar(task)
            if task.local:
                ks_ar.process_ks_ars()


    def get_kss(self, input_types=[], output_types=[], ignore_kss=[]):
        ignore_names = [ks.name for ks in ignore_kss]
        props = { hasonly(smile.hasInputDataLevels):input_types, 
                nothas(smile.hasName):ignore_names}
        if output_types: props[hasonly(smile.hasOutputDataLevels)] = output_types

        return Ks.search(props=props, how='all', subclass=False)


    def get_kss_for_hypo_improvement(self, hypothesis):
        hypothesis.load()
        ignore_names = [Ks(KSAR(ks_ar).ks).name for ks_ar in hypothesis.for_ks_ars]
        return Ks.search(props={smile.hasInputDataLevels:hypothesis.klass, smile.hasOutputDataLevels:hypothesis.klass, nothas(smile.hasName):ignore_names}, how='all', subclass=False)

    def get_kss_for_hypo_improvement_with_other_inputs(self, hypothesis, input_types, output_types=[]):
        hypothesis.load()
        output_types += [hypothesis.klass]
        ignore_kss = [Ks(KSAR(ks_ar).ks) for ks_ar in hypothesis.for_ks_ars]
        kss = self.get_kss(input_types=[hypothesis.klass]+input_types, output_types=output_types, ignore_kss=ignore_kss)
        return kss

    def get_kss_for_hypo_processing(self, hypothesis, output_types=[]):
        hypothesis.load()
        ignore_names = [Ks(KSAR(ks_ar).ks).name for ks_ar in hypothesis.for_ks_ars]
        props = {hasonly(smile.hasInputDataLevels):[hypothesis.klass], 
                nothas(smile.hasName):ignore_names}
        if output_types: props[hasonly(smile.hasOutputDataLevels)] = output_types
        return Ks.search(props=props, how='all', subclass=False)

    def get_kss_for_hypo_processing_with_other_inputs(self, hypothesis, input_types, output_types=[]):
        hypothesis.load()
        ignore_names = [Ks(KSAR(ks_ar).ks).name for ks_ar in hypothesis.for_ks_ars]
        props = {hasall(smile.hasInputDataLevels):[hypothesis.klass]+input_types, 
                nothas(smile.hasName):ignore_names}
        if output_types: props[hasonly(smile.hasOutputDataLevels)] = output_types
        kss = Ks.search(props=props, how='all', subclass=False)
        return kss




    def get_highest_ranked_local_hypo(self,subclass=False, input_type=None,  **kswars):
        hypotheses = self.collect_outputs()
        return self.get_highest_ranked_hypo(kswars, trace_id=self.trace.id, hypotheses=hypotheses, input_type=input_type)
    
    @classmethod
    def get_highest_ranked_global_hypo(cls,trace_id, subclass=False, input_type=None,  **kswars):
        if input_type is not None:
            hypotheses = input_type.search(props={smile.hasTraceID:trace_id}, how='all', subclass=subclass)
        else:
            hypotheses = smile.Hypothesis.search(props={smile.hasTraceID:trace_id}, how='all', subclass=True)
        # hypotheses = [hypo.cast_to_graph_type() for hypo in hypotheses]
        return cls.get_highest_ranked_hypo(kswars, hypotheses=hypotheses, input_type=input_type)

    @classmethod
    def get_highest_ranked_hypo(cls, label, hypotheses, input_type=None, from_kss=None):
        if from_kss is None:
            if input_type is None:
                from_kss = Ks.search(how='all', props={exists(smile.hasName):None}, subclass=False)
            else:
                from_kss = Ks.search(props={smile.hasInputDataLevels:input_type}, how='all', subclass=True)

        to_processes = {}
        # collect ksars that have not procesed each hypothesis in list
        for hypo in hypotheses:
            to_processes[hypo] = [ks.id for ks in from_kss.copy() if hypo.klass in ks.inputs]
            ksars = [KSAR(inst_id=ksar_id).cast_to_graph_type() for ksar_id in hypo.for_ks_ars]
            for ksar in ksars:
                # remove any ksars that already processed this 
                if str(ksar.ks) in [ks.id for ks in from_kss] and ksar.ks in to_processes[hypo]:
                    to_processes[hypo].remove(ksar.ks)

        tmp = [(i.certainty, i, kss) for i,kss in to_processes.items() if len(kss)>0]
        hypos_to_processes = [(i,kss) for c,i,kss in sorted(tmp, key=lambda x: (x[0],x[2]))]
        if len(hypos_to_processes) > 0:
            return hypos_to_processes[-1]
        else:
            return None, None
        
    def get_lowest_ranked_hypo(self, label, input_type, from_kss=None, subclass=False):
        if from_kss is None:
            from_kss = Ks.search(props={smile.hasInputDataLevels:input_type}, how='all', subclass=False)
        hypotheses = input_type.search(props={smile.hasTraceID:self.trace.id}, how='all', subclass=subclass)

        to_processes = {}
        # collect ksars that have not procesed each hypothesis in list
        for hypo in hypotheses:
            to_processes[hypo] = [ks.id for ks in from_kss.copy()]
            ksars = [KSAR(inst_id=ksar_id).cast_to_graph_type() for ksar_id in hypo.for_ks_ars]
            for ksar in ksars:
                # remove any ksars that already processed this 
                if str(ksar.ks) in [ks.id for ks in from_kss] and ksar.ks in to_processes[hypo]:
                    to_processes[hypo].remove(ksar.ks)

        tmp = [(i.certainty, i, kss) for i,kss in to_processes.items() if len(kss)>0]
        hypos_to_processes = [(i,kss) for c,i,kss in sorted(tmp, key=lambda x: (x[0],x[2]))]
        if len(hypos_to_processes) > 0:
            return hypos_to_processes[0]
        else:
            return None, None


    def subschedule_run(self, hypothesis, trigger_event, search):
        r1 = self.subschedule_improve(hypothesis=hypothesis, trigger_event=trigger_event, search=search, process_type=None)
        r2 = self.subschedule_process(hypothesis=hypothesis, trigger_event=trigger_event, search=search, process_type=None)
        return r1 + r2
    def subschedule_improve(self, hypothesis, trigger_event, search, process_type='improve', ):
        # print('\t\t5.1')
        schedule0 = self
        # improve hypothesis
        schedule1 = Schedule(trace=schedule0.trace, inputs=[hypothesis])
        # print('\t\t5.2')
        self.subschedules.append(schedule1)
        # print('\t\t5.3')
        schedule1.cycle = schedule0.cycle + 1
        # print('\t\t5.4')
        kss_to_add = schedule1.get_kss_for_hypo_improvement(hypothesis=hypothesis)
        # print('\t\t5.5')
        schedule1.processes_kss(kss=kss_to_add, search=search, process_type=process_type, trigger_event=f"{trigger_event}: Improve solo input")
        # print('\t\t5.6')

        # improve hypothesis with other hypotheses
        schedule2 = Schedule(trace=schedule0.trace, inputs=[hypothesis])
        # print('\t\t5.7')
        self.subschedules.append(schedule2)
        # print('\t\t5.8')
        schedule2.cycle = schedule0.cycle + 1
        # print('\t\t5.9')
        kss_to_add = schedule2.get_kss_for_hypo_improvement_with_other_inputs(hypothesis=hypothesis, input_types=[])
        # print('\t\t5.10')
        schedule2.processes_kss(kss=kss_to_add, search=search, process_type=process_type, trigger_event=f"{trigger_event}: Improve paired input")
        # print('\t\t5.11')

        return schedule1, schedule2

    def subschedule_process(self, hypothesis, trigger_event, search, process_type='process'):
        s1 = self.subschedule_process0(hypothesis=hypothesis, trigger_event=trigger_event, search=search, process_type=process_type)
        s2 = self.subschedule_process1(hypothesis=hypothesis, trigger_event=trigger_event, search=search, process_type=process_type)
        return s1, s2
    def subschedule_process0(self, hypothesis, trigger_event, search, process_type='process'):
        schedule0 = self
        # processes hypothesis
        schedule1 = Schedule(trace=schedule0.trace, inputs=[hypothesis])
        self.subschedules.append(schedule1)
        schedule1.cycle = schedule0.cycle + 1
        kss_to_add = schedule1.get_kss_for_hypo_processing(hypothesis=hypothesis)
        schedule1.processes_kss(kss=kss_to_add, search=search, process_type=process_type, trigger_event=f"{trigger_event}: Process solo input")
        return schedule1

    def subschedule_process1(self, hypothesis, trigger_event, search, process_type='process'):
        # processes hypothesis with other hypotheses
        schedule0 = self
        schedule2 = Schedule(trace=schedule0.trace, inputs=[hypothesis])
        self.subschedules.append(schedule2)
        schedule2.cycle = schedule0.cycle + 1
        kss_to_add = schedule2.get_kss_for_hypo_processing_with_other_inputs(hypothesis=hypothesis, input_types=[])
        schedule2.processes_kss(kss=kss_to_add, search=search, process_type=process_type, trigger_event=f"{trigger_event}: Process paired input")

        return schedule2

    def processes_kss(self, **kwargs):
        if kwargs['search']=='dfs':
            return self.processes_kss_dfs(**kwargs)
        elif kwargs['search']=='bfs':
            return self.processes_kss_bfs(**kwargs)

    def processes_kss_dfs(self, kss, search, process_type, trigger_event, **kwargs):
        for ks in kss:
            task = Task(ks)
            task.inputs = self.inputs
            ks_ar = self.gen_ksar(task=task, trigger_event=trigger_event)
            self.tasks.append(task)

        tmp_outputs = []
        for task in self.tasks:
            task.run()
            ks_ar = task.ks_ar
            if ks_ar is not None:
                ks_ar.load()

                outs = [Hypothesis(inst_id=hypo_id).cast_to_graph_type() for hypo_id in ks_ar.hypotheses]
                # get top result
                outs = sorted(outs, key=lambda x: x.certainty, reverse=True)
                tmp_outputs += outs
                print("\t"*self.cycle, f"dfs({search}) = {self.id}/{self.cycle} process_kss: outs", len(outs), [out.klass for out in outs])
                for hypothesis in outs:
                    if process_type is None or process_type=='improve':
                        s1,s2 = self.subschedule_improve(hypothesis, search=search, trigger_event=f"Received new 2")
                        print("\t"*self.cycle, f"{self.id}/{self.cycle} subimprov: {s1.id, s2.id}")
                    if process_type is None or process_type=='process':
                        s3,s4 = self.subschedule_process(hypothesis, search=search, trigger_event=f"Received new 3")
                        print("\t"*self.cycle, f"{self.id}/{self.cycle} subimproc: {s3.id, s4.id}")
        self.outputs += tmp_outputs
    def processes_kss_bfs(self, kss, search, process_type, trigger_event, **kwargs):
        for ks in kss:
            task = Task(ks)
            task.inputs = self.inputs
            ks_ar = self.gen_ksar(task=task, trigger_event=trigger_event)
            self.tasks.append(task)

        tmp_outputs = []
        for task in self.tasks:
            task.run()
            ks_ar = task.ks_ar
            if ks_ar is not None:
                ks_ar.load()

                outs = [Hypothesis(inst_id=hypo_id).cast_to_graph_type() for hypo_id in ks_ar.hypotheses]
                # get top result
                outs = sorted(outs, key=lambda x: x.certainty, reverse=True)
                tmp_outputs += outs
                print("\t"*self.cycle, f"bfs({search}) = {self.id}/{self.cycle} process_kss: outs", len(outs), [out.klass for out in outs])
        self.outputs += tmp_outputs


    @classmethod
    def set_kss(cls):
        ######################################################################
        # Parse Query
        ######################################################################

        kss = Ks.search(props={smile.hasPyName:'ParseQuery'}, how='all')
        for ks in kss:
            ks.delete(refs=False)
        ALL_KS_FORMATS = {
            'Parse/Query': ['ParseQuery', False, ['Query'], ['Text']],
        }

        for ks_name, fields in ALL_KS_FORMATS.items():
            Ks.ALL_KS_FORMATS[ks_name] = fields
        for ks_name in ALL_KS_FORMATS.keys():
            Ks.initialize_ks(ks_name)

        ######################################################################
        # text to Sentences
        ######################################################################
        kss = Ks.search(props={smile.hasPyName:'TextToSentence'}, how='all')
        for ks in kss:
            ks.delete(refs=False)

        ALL_KS_FORMATS = {
            'Text To Sentences': ['TextToSentence', False, ["Text"], ["Sentence","Word"]],
        }

        for ks_name, fields in ALL_KS_FORMATS.items():
            Ks.ALL_KS_FORMATS[ks_name] = fields
        for ks_name in ALL_KS_FORMATS.keys():
            Ks.initialize_ks(ks_name)



        ######################################################################
        # ChatGPT Prompt 0 / 1
        ######################################################################
        kss = Ks.search(props={hasany(smile.hasPyName):['ChatGPTPrompt0','ChatGPTPrompt1']}, how='all')
        for ks in kss:
            ks.delete(refs=False)

        ALL_KS_FORMATS = {}
        for klass0 in  ['Service', 'Program', 'Organization', 'BeneficialStakeholder', "Outcome"]:
            ALL_KS_FORMATS[f'ChatGPTPrompt0 (Text)({klass0})'] = ['ChatGPTPrompt0', False, ["Text"], [klass0]]
            # ALL_KS_FORMATS[f'ChatGPTPrompt0 (Sentence)({klass0})'] = ['ChatGPTPrompt0', False, ["Sentence"], [klass0]]

        for klass0 in ['Service', 'Program', 'Organization', 'BeneficialStakeholder', "Outcome"]:
            for klass1 in  ['Service', 'Program', 'Organization', 'BeneficialStakeholder', "Outcome"]:
                if klass0 == klass1:
                    continue
                ALL_KS_FORMATS[f'ChatGPTPrompt1 (Text,{klass0})({klass1})'] = ['ChatGPTPrompt1', False, ["Text", klass0], [klass1]]
                # ALL_KS_FORMATS[f'ChatGPTPrompt1 (Sentence,{klass0})({klass1})'] = ['ChatGPTPrompt1', False, ["Sentence", klass0], [klass1]]

        for ks_name, fields in ALL_KS_FORMATS.items():
            Ks.ALL_KS_FORMATS[ks_name] = fields
        for ks_name in ALL_KS_FORMATS.keys():
            Ks.initialize_ks(ks_name)



        ##############################################################################
        # QA 0
        ##############################################################################
        kss = Ks.search(props={smile.hasPyName:'Qa0Ner'}, how='all')
        for ks in kss:
            ks.delete(refs=False)
        ALL_KS_FORMATS = {}
        # ALL_KS_FORMATS['QA-0 (Text)(Program)'] = ['Qa0Ner', False, ["Text"], ["Program"]]
        ALL_KS_FORMATS['QA-0 (Sentence)(Program)'] = ['Qa0Ner', False, ["Sentence"], ["Program"]]
        # ALL_KS_FORMATS['QA-0 (Text)(Outcome)'] = ['Qa0Ner', False, ["Text"], ["Outcome"]]
        ALL_KS_FORMATS['QA-0 (Sentence)(Outcome)'] = ['Qa0Ner', False, ["Sentence"], ["Outcome"]]
        # ALL_KS_FORMATS['QA-0 (Text)(BeneficialStakeholder)'] = ['Qa0Ner', False, ["Text"], ["BeneficialStakeholder"]]
        ALL_KS_FORMATS['QA-0 (Sentence)(BeneficialStakeholder)'] = ['Qa0Ner', False, ["Sentence"], ["BeneficialStakeholder"]]
        for ks_name, fields in ALL_KS_FORMATS.items():
            Ks.ALL_KS_FORMATS[ks_name] = fields
        for ks_name in ALL_KS_FORMATS.keys():
            Ks.initialize_ks(ks_name)

        ##############################################################################
        # QA 1
        ##############################################################################            
        kss = Ks.search(props={smile.hasPyName:'Qa1Ner'}, how='all')
        for ks in kss:
            ks.delete(refs=False)

        ALL_KS_FORMATS = {}
        # ALL_KS_FORMATS['QA-1 (Program,Text)(BeneficialStakeholder)'] = ['Qa1Ner', False, ["Text", "Program"], ["BeneficialStakeholder"]]
        ALL_KS_FORMATS['QA-1 (Program,Sentence)(BeneficialStakeholder)'] = ['Qa1Ner', False, ["Sentence", "Program"], ["BeneficialStakeholder"]]
        # ALL_KS_FORMATS['QA-1 (BeneficialStakeholder,Text)(Program)'] = ['Qa1Ner', False, ["Text", "BeneficialStakeholder"], ["Program"]]
        ALL_KS_FORMATS['QA-1 (BeneficialStakeholder,Sentence)(Program)'] = ['Qa1Ner', False, ["Sentence", "BeneficialStakeholder"], ["Program"]]
        # ALL_KS_FORMATS['QA-1 (Program,Text)(Outcome)'] = ['Qa1Ner', False, ["Text", "Program"], ["Outcome"]]
        ALL_KS_FORMATS['QA-1 (Program,Sentence)(Outcome)'] = ['Qa1Ner', False, ["Sentence", "Program"], ["Outcome"]]
        # ALL_KS_FORMATS['QA-1 (Outcome,Text)(BeneficialStakeholder)'] = ['Qa1Ner', False, ["Text", "Outcome"], ["BeneficialStakeholder"]]
        ALL_KS_FORMATS['QA-1 (Outcome,Sentence)(BeneficialStakeholder)'] = ['Qa1Ner', False, ["Sentence", "Outcome"], ["BeneficialStakeholder"]]
        for ks_name, fields in ALL_KS_FORMATS.items():
            Ks.ALL_KS_FORMATS[ks_name] = fields
        for ks_name in ALL_KS_FORMATS.keys():
            Ks.initialize_ks(ks_name)

