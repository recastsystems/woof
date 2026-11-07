"""Explicit loops for the small array-expression subset used by CAM columns."""
import re
from make_driver import split


def bounds(text,default):
    source=text if ':' in text else default
    lo,hi=source.split(':',1) if ':' in source else ('1',source)
    if not lo:lo=default.split(':')[0] if ':' in default else '1'
    if not hi:hi=default.split(':')[1] if ':' in default else default
    return lo,hi


def calls(text):
    result=[]
    pattern=re.compile(r'\b(\w+)\(')
    position=0
    while match:=pattern.search(text,position):
        depth=1;end=match.end()
        while depth and end<len(text):
            depth+=(text[end]=='(')-(text[end]==')');end+=1
        result.append((match.start(),end,match[1],split(text[match.end():end-1])))
        position=end
    return result


def explicit(statements,shapes):
    result=[];variables=set();serial=0
    for stmt in statements:
        if stmt.startswith(('subroutine','function','real','integer','logical','parameter','data','save','implicit','end subroutine','end function')):
            result.append(stmt);continue
        counted=re.search(r'count\((\w+)\(([^)]+)\)\)',stmt)
        if counted:
            serial+=1;counter='cam_count_'+str(serial);variables.update((counter,'cam_red_k'))
            array,section=counted.groups();lo,hi=bounds(section,shapes[array][0])
            result.extend([counter+'=0','do cam_red_k='+lo+','+hi,'if('+array+'(cam_red_k)) '+counter+'='+counter+'+1','end do'])
            stmt=stmt[:counted.start()]+counter+stmt[counted.end():]
        if 'maxval(' in stmt:
            if not re.fullmatch(r'maxcld\(1:ncol\)\s*=\s*maxval\(cldp\(1:ncol,ntoplw:pver\),dim=2\)',stmt):
                raise ValueError('unrecognized CAM max reduction: '+stmt)
            variables.update(('cam_red_i','cam_red_k'))
            result.extend(['do cam_red_i=1,ncol','maxcld(cam_red_i)=cldp(cam_red_i,ntoplw)',
                           'do cam_red_k=ntoplw+1,pver','maxcld(cam_red_i)=max(maxcld(cam_red_i),cldp(cam_red_i,cam_red_k))','end do','end do'])
            continue
        if stmt.startswith('call '):
            # Contiguous one-dimensional argument sections use sequence association.
            def section(match):
                array,indices=match.groups()
                if array not in shapes or ':' not in indices:return match[0]
                items=split(indices)
                for axis,item in enumerate(items):
                    if ':' in item:
                        lo,hi=bounds(item,shapes[array][axis]);items[axis]=lo
                return array+'('+','.join(items)+')'
            stmt=re.sub(r'\b(\w+)\(([^()]*)\)',section,stmt)
            if ':' in stmt:raise ValueError('unrecognized CAM argument section: '+stmt)
            result.append(stmt);continue
        product=re.fullmatch(r'(\w+)\s*=\s*product\((\w+)\(([^)]+)\)\)',stmt)
        if product:
            target,array,section=product.groups();lo,hi=bounds(section,shapes[array][0]);variables.add('cam_red_k')
            result.extend([target+'=1','do cam_red_k='+lo+','+hi,target+'='+target+'*'+array+'(cam_red_k)','end do'])
            continue
        summed=re.fullmatch(r'(\w+)\s*=\s*sum\((\w+)\)',stmt)
        if summed:
            target,array=summed.groups();lo,hi=bounds(':',shapes[array][0]);variables.add('cam_red_k')
            result.extend([target+'=0.0_r8','do cam_red_k='+lo+','+hi,target+'='+target+'+'+array+'(cam_red_k)','end do'])
            continue
        whole=re.match(r'^(\w+)\s*=(?!=)\s*(.*)$',stmt)
        if whole and whole[1] in shapes:
            stmt=whole[1]+'('+','.join(':' for _ in shapes[whole[1]])+')='+whole[2]
        match=re.match(r'^(\w+)\((.*?)\)\s*=(?!=)\s*(.*)$',stmt)
        if not match or ':' not in match[2]:
            result.append(stmt);continue
        name,indices,rhs=match.groups();lhs=split(indices)
        axes=[i for i,arg in enumerate(lhs) if ':' in arg]
        counters=[];loops=[]
        serial+=1
        for axis in axes:
            counter=f'cam_s{serial}_{axis}';variables.add(counter);counters.append(counter)
            lo,hi=bounds(lhs[axis],shapes[name][axis])
            loops.append('do '+counter+'=0,('+hi+')-('+lo+')')
            lhs[axis]='('+lo+'+'+counter+')'
        edits=[]
        for start,end,array,items in calls(rhs):
            if array not in shapes:continue
            sliced=[i for i,arg in enumerate(items) if ':' in arg]
            if sliced and len(sliced)!=len(counters):raise ValueError('nonconforming array ranks: '+stmt)
            for axis,counter in zip(sliced,counters):
                lo,_=bounds(items[axis],shapes[array][axis]);items[axis]='('+lo+'+'+counter+')'
            edits.append((start,end,array+'('+','.join(items)+')'))
        for start,end,replacement in reversed(edits):rhs=rhs[:start]+replacement+rhs[end:]
        for array,shape in shapes.items():
            pattern=r'\b'+array+r'\b(?!\s*\()'
            if re.search(pattern,rhs):
                if len(shape)!=len(counters):raise ValueError('nonconforming whole array: '+stmt)
                items=[bounds(':',dimension)[0]+'+'+counter for dimension,counter in zip(shape,counters)]
                rhs=re.sub(pattern,array+'('+','.join(items)+')',rhs)
        result.extend(loops+[name+'('+','.join(lhs)+')='+rhs]+['end do']*len(loops))
    # F77 labels preserve the modern cycle and exit semantics of the innermost loop.
    normalized=[];stack=[];label=9000;selectors=[]
    for stmt in result:
        select=re.fullmatch(r'select\s+case\s*\((\w+)\)',stmt)
        if select:selectors.append([select[1],False]);continue
        case=re.fullmatch(r'case\s*\(([^)]+)\)',stmt)
        if case:
            selector,used=selectors[-1]
            condition='.or.'.join(selector+'=='+v for v in split(case[1]))
            normalized.append(('else if(' if used else 'if(')+condition+') then');selectors[-1][1]=True;continue
        if stmt=='case default':normalized.append('else');continue
        if stmt=='end select':selectors.pop();normalized.append('endif');continue
        if stmt=='do':stmt='do while(.true.)'
        if re.match(r'^do\s',stmt):
            label+=2;stack.append((label,label+1))
        if re.fullmatch(r'end\s*do|enddo',stmt):
            end,after=stack.pop();normalized.extend([str(end)+' continue',stmt,str(after)+' continue']);continue
        if re.search(r'\b(?:cycle|exit)\b',stmt):
            if not stack:raise ValueError('loop control outside loop: '+stmt)
            stmt=re.sub(r'\bcycle\b','goto '+str(stack[-1][0]),stmt)
            stmt=re.sub(r'\bexit\b','goto '+str(stack[-1][1]),stmt)
        normalized.append(stmt)
    return normalized,variables
