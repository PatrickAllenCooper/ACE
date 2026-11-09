from pathlib import Path
p=Path('/tmp/ace-downstream-compact-d1d2-1d_57cvu/implementation/compact_inputs.py');s=p.read_text()
new='''
class OperationEvidence:
    """Durable caller-owned records; reference is published before any transport."""
    def __init__(self, root, identity, publish):
        self.root=Path(root);self.publish=publish;self.number=0;self.last_reference=None
        need(self.root.parent==self.root.parent.resolve(strict=True),'canonical durable evidence parent required')
        self.root.mkdir(mode=0o700,exist_ok=False)
        self.intent={'schema':'ace-compact-input-use-intent-v1','path':str(self.root),
            'identity':identity,'acceptance':False,'retry_allowed':False}
        self.intent_raw=canonical(self.intent)
        self._write('intent.json',self.intent_raw)
        self.intent_reference={'path':str(self.root),'intent_sha256':sha(self.intent_raw),
                               'acceptance':False,'retry_allowed':False}
        # If publication fails, no backend method has run. Original intent stays.
        self.publish('storage-use.intent.json',self.intent_reference)

    def _write(self,name,data):
        with (self.root/name).open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())

    def record(self, operation, error=None, details=None):
        ordinal=self.number;self.number+=1
        path=self.root/'operations'
        view=selected_inventory(path) if path.exists() else {'files':{},'directories':[]}
        manifest={'schema':'ace-compact-input-operation-inventory-v1',
            'intent_sha256':sha(self.intent_raw),'operation':operation,'ordinal':ordinal,
            'status':'completed' if error is None else 'failed',
            'error':None if error is None else {'type':type(error).__name__,'message':str(error)},
            'details':details,'operations_path':str(path),'inventory':view,'acceptance':False,'retry_allowed':False}
        data=canonical(manifest);name=f'inventory-{ordinal:03d}.json';self._write(name,data)
        reference={**self.intent_reference,'inventory_path':str(self.root/name),
                   'inventory_sha256':sha(data),'status':manifest['status'],'operation':operation}
        self._write(f'reference-{ordinal:03d}.json',canonical(reference))
        self.last_reference=reference
        # Always retain the original manifest/reference if caller publication fails.
        self.publish(f'storage-use-{ordinal:03d}.json',reference)
        return reference

    def run(self, operation, action, details=None):
        try:
            result=action()
        except BaseException as primary:
            try:self.record(operation,primary,details)
            except BaseException as secondary:
                try:self._write(f'evidence-failure-{self.number:03d}.json',canonical({
                    'primary':{'type':type(primary).__name__,'message':str(primary)},
                    'secondary':{'type':type(secondary).__name__,'message':str(secondary)},
                    'acceptance':False,'retry_allowed':False}))
                except BaseException:pass
            raise
        self.record(operation,details=details)
        return result


def recorded(operation):
    # Direct standalone Inputs tests may omit evidence; production bind cannot.
    def decorate(function):
        def run(self,*args,**kwargs):
            if self.evidence is None:return function(self,*args,**kwargs)
            details={'member':args[0],'expected_sha256':args[1]} if operation=='input-read' and len(args)>1 else None
            return self.evidence.run(operation,lambda:function(self,*args,**kwargs),details)
        return run
    return decorate


'''
s=s.replace('class Inputs:\n',new+'class Inputs:\n')
s=s.replace('expected_roots, helper_sha256, local_roots, staging, helper_path):','expected_roots, helper_sha256, local_roots, staging, helper_path, evidence=None):')
s=s.replace('        self.used={};self.captured={};self.attempted=set();self.ordinal=0','        self.evidence=evidence\n        self.used={};self.captured={};self.attempted=set();self.ordinal=0')
s=s.replace('    def recheck(self):','    @recorded("archive-verify")\n    def recheck(self):')
s=s.replace('    def read(self, name, expected_sha256, maximum=256*1024**2):','    @recorded("input-read")\n    def read(self, name, expected_sha256, maximum=256*1024**2):')
s=s.replace('    import tempfile\n    import types','    import types')
s=s.replace("    parent=Path(tempfile.mkdtemp(prefix='ace-compact-input-use-',dir='/tmp')).resolve()", "    evidence=ctx['storage_operation_evidence']\n    parent=evidence.root\n    need(evidence.last_reference is None,'fresh registered predecessor-use scope required')")
s=s.replace("local_roots=[p/'remote' for p in capsules],staging=parent/'operations',helper_path=helper_path)","local_roots=[p/'remote' for p in capsules],staging=parent/'operations',helper_path=helper_path,evidence=evidence)")
# Original post-hoc unpinned use.json replaced by a pinned completion inventory.
start=s.index("    (parent/'use.json').write_bytes");end=s.index("    ctx['compact_inputs']=gate",start)
s=s[:start]+"    evidence.record('binding-complete',details={'predecessor':predecessor,'receipt_sha256':receipt_pin,'ancestry':chain})\n"+s[end:]
p.write_text(s)
