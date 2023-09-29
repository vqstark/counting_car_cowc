def hyp_parse(hyp_path):
    hyp = {}
    keys = [] 
    with open(hyp_path,'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip())==0 : continue
            v = line.strip().split(':')
            try:
                hyp[v[0]] = float(v[1].strip().split(' ')[0])
            except:
                hyp[v[0]] = eval(v[1].strip().split(' ')[0])
            keys.append(v[0])
        f.close()
    return hyp

def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report == 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))