load image_transformation_DA_newdata
fs=10000;
level=3;
x=data(1,:);
[~, ~, ~, fc, ~, BW] = kurtogram(x, fs, level);
bpf = designfilt('bandpassfir', 'FilterOrder', 200, 'CutoffFrequency1', fc-BW/2, ...
    'CutoffFrequency2', fc+BW/2, 'SampleRate', fs);
xOuterBpf = filter(bpf, x);
[pEnvOuterBpf, fEnvOuterBpf, xEnvOuterBpf, tEnvBpfOuter] = envspectrum(x, fs, ...
    'FilterOrder', 200, 'Band', [fc-BW/2 fc+BW/2]);

