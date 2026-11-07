"""Restore symbolic local-array extents after the constrained F77 conversion.

Only array declarations and their dimension-derived index constants are edited.
The original arithmetic statements and table dimensions remain unchanged.
"""
import json,re
from pathlib import Path

VALUES={'pcols':'1','ncol':'1','pver':'CAM_LEVELS','pverp':'(CAM_LEVELS+1)','pverp2':'(CAM_LEVELS+2)','pverp3':'(CAM_LEVELS+3)','pverp4':'(CAM_LEVELS+4)','pverr':'CAM_LEVELS','pverrp':'(CAM_LEVELS+1)','kte':'CAM_LEVELS','kts':'1','kme':'(CAM_LEVELS+1)','kms':'1','ite':'1','its':'1','ims':'1','ime':'1','jms':'1','jme':'1','levsiz':'59','num_months':'13','ppcnst':'3','pcnst':'3','paerlev':'29','naer_c':'13','n_cldadv':'3','cam_abs_dim1':'4','cam_abs_dim2':'(CAM_LEVELS+1)','nbands':'2','bnd_nbr_lw':'7','naer_all':'13','nspint':'19','naer_groups':'7','nsubcol':'15','nconfgmax':'15','ndstsz':'4','naer':'11'}

def symbolic(text):
 return re.sub(r'\b[a-z]\w*\b',lambda m:VALUES.get(m[0],m[0]),text)

def value(text):
 return eval(text,{'__builtins__':{}},{'CAM_LEVELS':32})

def geometry(shape):
 bounds=[];lengths=[]
 for dim in shape:
  dim=symbolic(dim);lo,hi=dim.split(':') if ':' in dim else ('1',dim)
  bounds.append(lo);lengths.append('('+hi+'-('+lo+')+1)')
 return bounds,lengths

def count_expression(shape):
 return '*'.join(geometry(shape)[1]) or '1'

def outside_nested(expression,mapping):
 result=[];depth=0;start=0
 for i,ch in enumerate(expression):
  if ch=='[' and depth==0:
   result.append(replace_numbers(expression[start:i],mapping));start=i
  depth+=(ch=='[')-(ch==']')
  if ch==']' and depth==0:
   result.append(expression[start:i+1]);start=i+1
 result.append(replace_numbers(expression[start:],mapping))
 return ''.join(result)

def replace_numbers(text,mapping):
 # F2C reduces power-of-two array strides to shifts. Restore those too.
 text=re.sub(r'<<\s*(\d+)',lambda m:'* ('+mapping[1<<int(m[1])]+')' if (1<<int(m[1])) in mapping else m[0],text)
 return re.sub(r'(?<![\w.])\d+(?![\w.])',lambda m:'('+mapping[int(m[0])]+')' if int(m[0]) in mapping else m[0],text)

def local_arrays(code,layout):
 for original,field in layout.items():
  shape=field['shape'];count=count_expression(shape)
  if 'CAM_LEVELS' not in count:continue
  name=original+'__' if '_' in original else original
  # F2C renames a few single-letter names to avoid C identifiers.
  found=re.search(r'\b'+name+r'\s*\[\s*'+str(value(count))+r'\s*\]',code)
  if not found:
   if not re.search(r'\b'+name+r'\b',code):continue
   raise ValueError('missing CAM local array declaration: '+original)
  lo,lengths=geometry(shape);stride='1';terms=[];mapping={}
  for lower,length in zip(lo,lengths):
   if 'CAM_LEVELS' in stride:
    mapping[value(stride)]=stride
    # F2C also folds fixed-coordinate multiples of a symbolic stride.
    if 'CAM_LEVELS' not in length:
     for coordinate in range(1,value(length)+1):
      mapping[coordinate*value(stride)]='('+str(coordinate)+')*('+stride+')'
   terms.append('('+lower+')*('+stride+')');stride='('+stride+')*('+length+')'
  offset='+'.join(terms)
  if 'CAM_LEVELS' in offset:mapping[value(offset)]=offset
  # Index constants belong to this array, not nested array accesses.
  pattern=re.compile(r'\b'+name+r'\s*\[');matches=[]
  for match in pattern.finditer(code):
   start=match.end();end=start;depth=1
   while depth:
    depth+=(code[end]=='[')-(code[end]==']');end+=1
   expression=code[start:end-1]
   replacement=count if match.start()==found.start() else outside_nested(expression,mapping)
   matches.append((start,end-1,replacement))
  for start,end,replacement in reversed(matches):code=code[:start]+replacement+code[end:]
 return code


def external_arrays(code,layout):
    expressions=[]
    for original,field in layout.items():
        count=count_expression(field['shape'])
        if 'CAM_LEVELS' not in count:continue
        name=original+'__' if '_' in original else original
        pattern=r'\b'+name+r'\[([^]]+)\]'
        found=re.search(pattern,code)
        if not found:continue
        # The first occurrence is the local declaration, before executable code.
        before=code[:found.start()];dtype={'double precision':'doublereal','real':'real','integer':'integer','logical':'logical'}[field['dtype']]
        replacement='*'+name+' = cam_alloc<'+dtype+'>(state, '+count+')'
        end=found.end()
        initializer=re.match(r'\s*=\s*\{\}',code[end:])
        if initializer:end+=initializer.end()
        code=code[:found.start()]+replacement+code[end:]
        expressions.append('(('+count+')*'+str(8 if dtype=='doublereal' else 4)+'+7)')
    if expressions:
        brace=code.index('{')+1;code=code[:brace]+'\n CamScratchFrame scratch_frame(state);\n'+code[brace:]
    return code,expressions


def explicit_column_frames(source, metadata):
    """Put mutable fixed arrays and per-worker module scalars in explicit GPU memory.

    Literal coefficient arrays keep their initializers. Scratch frames restore
    the arena cursor across calls, including calls inside the layer-pair loops.
    The budget sums all declarations, so it bounds every possible call chain.
    """
    pattern=re.compile(r'__device__\s+(?:int|integer|logical|doublereal)\s+\w+\(CamState\* state,.*?\)\s*\{',re.S)
    changes=[]; extra=0
    for match in pattern.finditer(source):
        start=match.end(); end=start; depth=1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}'); end+=1
        body=source[start:end-1]; allocated=[]
        def declaration(decl):
            dtype, entities=decl.groups()
            # A semicolon declaration can contain several entities and calls.
            # Only a numeric fixed extent with no initializer or {} is mutable.
            def array(item):
                name,count=item.group(1),int(item.group(2)); initializer=item.group(3)
                if initializer and initializer.strip()!='{}': return item[0]
                allocated.append(count*(8 if dtype=='doublereal' else 4)+7)
                return '*'+name+' = cam_alloc<'+dtype+'>(state, '+str(count)+')'
            entities=re.sub(r'\b(\w+)\[\s*(\d+)\s*\](?:\s*=\s*(\{[^}]*\}))?',array,entities)
            return '\n    '+dtype+' '+entities+';'
        body=re.sub(r'(?m)^\s*(doublereal|integer|real|logical)\s+([^;]+);',declaration,body)
        if allocated:
            if 'CamScratchFrame scratch_frame(state);' not in body:
                body='\n CamScratchFrame scratch_frame(state);\n'+body
            changes.append((start,end-1,body)); extra+=sum(allocated)
    for start,end,body in reversed(changes): source=source[:start]+body+source[end:]
    # A separate aligned slot belongs to each worker. The module table pointers
    # are shared read-only; all mutable scalar fields remain column-private.
    extra=(extra+7)&~7
    state_bytes=2048
    source=source.replace('CamState column_state = {};\nCamState* state=&column_state;',
        'CamState* state=(CamState*)(workspace+CAM_WORK_BYTES*CAM_FRAME_INDEX);')
    # Driver uses bounded workers; the three low-level replay entries use columns.
    for entry in ('cam_emissivity','cam_lw','cam_sw','cam_driver'):
        pos=source.index('void '+entry+'('); end=source.find('extern "C" __global__',pos+1)
        if end<0:end=len(source)
        part=source[pos:end].replace('CAM_FRAME_INDEX','worker' if entry=='cam_driver' else 'column')
        part=re.sub(r'state->scratch=workspace\+(worker|column)\*CAM_WORK_BYTES;',
            r'state->scratch=workspace+\1*CAM_WORK_BYTES+2048;',part)
        source=source[:pos]+part+source[end:]
    source=source.replace('#define CAM_WORK_BYTES (','#define CAM_WORK_BYTES ('+str(state_bytes+extra)+'+',1)
    source=source.replace('struct CamScratchFrame{','static_assert(sizeof(CamState)<=2048,"CAM private state slot");\nstruct CamScratchFrame{',1)
    for entry in metadata.values():
        entry['workspace_expression']='('+str(state_bytes+extra)+'+'+entry['workspace_expression']+')'
    return source, metadata
