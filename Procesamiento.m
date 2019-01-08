%% Carga las variables generadas por la app y carga las rutas de los audios

load('Procesconfig.mat')
datafolder = data.processdir;
records = audioDatastore(datafolder, 'IncludeSubfolders', true,...
                         'FileExtensions', '.wav', 'LabelSource',...
                         'foldernames');
                 
%% Agrupa y etiqueta los audios que cada red reconocerá

NumCNN = 3; % Número de redes a entrenar, para generalizar el script esta
% variable se puede volver dependiente del usuario, es igual al número de
% celdas con el conjunto de las palabras a reconocer por cada red
organized = cell([NumCNN, 2]); 
organized{1,1} = categorical(["Gato", "Pera", "Mano", "Fresa", "Vaca",...
                              "Codo", "Sol", "Ventana", "El", "Niño",...
                              "Llora", "Hombre", "Camina", "Lentamente",...
                              "Por", "La", "Calle"]);
organized{2,1} = categorical(["León", "Caballo", "Toro", "Perro", "Cabra",...
                              "Pájaro", "Pollo", "Cerdo", "Paloma", "Rata",...
                              "Cordero", "Gallo", "Ratón", "Mono", "Pez",...
                              "Gallina", "Serpiente", "Tigre", "Oso",...
                              "Canario"]);
organized{3,1} = categorical(["Fué", "Forma", "Fueron", "Frente", "Fin",...
                              "Fuera", "Final", "Familia", "Falta",...
                              "Fuerza", "Fondo", "Futuro", "Figura",...
                              "Fuerte", "Fuente", "Función", "Favor",...
                              "Formación", "Fecha", "Fácil"]);

CNNin = cell([NumCNN 3]); % el 3 viene de los 3 conjuntos de entrenamiento,
PCNNin = CNNin;           % validación y test; PCNNin = Processed CNNin
includeFraction = 0.2;

for i = 1:NumCNN
organized{i,2} = audiorganize(records, organized{i,1}, includeFraction);
[CNNin{i,1}, CNNin{i,2},CNNin{i,3}] = divideData(organized{i,2});
end

%% Obtención del espectrograma con las especificaciones hechas por la app

disp("Extrayendo características del audio...");
numBands = size(nFilterBank, 2); numHops = size(Idx, 2);

data = struct; data.('filtro') = filtro; data.('Idx') = Idx; 
data.('nfft') = nfft; data.('nFilterBank') = nFilterBank; 
data.('pWindow') = pWindow;data.('numBands') = numBands; 
data.('numHops') = numHops;

for i = 1:NumCNN
    for j = 1:3
        PCNNin{i,j} = Process(CNNin{i,j}, data);
    end 
end

%% Grafica el espectrograma, voz, reproduce sonido y la palabra asociada
% Se muestran para corroborar que esté correcto el script

duracion = 1.5;
specMin = min(min(min(PCNNin{1,1}(:,:,:))));
specMax = max(max(max(PCNNin{1,1}(:,:,:))));
idx = randperm(size(PCNNin{1,1}(:,:,:),3),3);
figure('Units', 'normalized', 'Position',[0.2 0.2 0.6 0.6]);
for i = 1:NumCNN
    [x,fs] = audioread(string(CNNin{1,1}.Files(idx(i))));
    subplot(2,3,i)
    t = linspace(0, duracion, numel(x));
    plot(t, x)
    axis tight
    title(string(CNNin{1,1}.Labels(idx(i))))

    subplot(2,3,i+3)
    spect = PCNNin{1,1}(:,:,idx(i));
    pcolor(spect)
    caxis([specMin+2 specMax])
    shading flat

    sound(x,fs)
    pause(2)
end

%% Visualiza la distribución de pixeles en de los espectrogramas
% ¿Podría usarse para estimar si se tiene el número suficiente de muestras?

h = figure('Units', 'normalized');
h.WindowState = 'maximized';
formatSpec = 'Distribución de las entradas de la neurona %d';
h = gobjects(3,1);
for i = 1:NumCNN
    h(i) = subplot(2,2,i);
    histogram(PCNNin{1,i},'EdgeColor','none','Normalization','pdf')
    axis tight
    ax = gca;
    ax.YScale = 'log';
    xlabel("Valor del pixel en los espectrogramas")
    ylabel("Densidad de Probabilidad")
    grid on
    title(sprintf(formatSpec, i));
end
pos = get(h,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h(NumCNN),'Position',[new,pos{end}(2:end)])

%% Ventana donde se pregunta al usuario si se incluyen archivos de ruido

respuesta = ''; % Carácter vacío
opts.Interpreter = 'tex';
opts.Default = 'Volver';

while isempty(respuesta)
    respuesta = questdlg('\fontsize{10} ¿Quieres agregar archivos de ruido al entrenamiento de la red?',...
                         'Agregar ruido', 'No agregar', 'No agregar', opts);
    if strcmpi(respuesta, 'Agregar ruido')
        disp('Comienzan a añadirse los archivos de ruido')
        background(trained)
        
    elseif strcmpi(respuesta, 'No agregar')
        disp('No se agregaron los archivos de ruido');
    end 
end

%% Visualiza la distribución de grabaciones por cada red

h = figure('Units','normalized');
h.WindowState = 'maximized';
formatSpec = 'Palabras para entrenamiento en la red %d';
formatSpec1 = 'Palabras para validación en la red %d';
h = gobjects(6,1);
for i = 1:NumCNN 
    subplot(2,3,0+i)
    histogram(CNNin{i,1}.Labels)
    title(sprintf(formatSpec, i))
    subplot(2,3,3+i)
    histogram(CNNin{i,2}.Labels)
    title(sprintf(formatSpec, i))
end

%% Ventana donde el usuario decide si quiere continuar con el entrenamiento

respuesta = ''; % Carácter vacío
opts.Interpreter = 'tex';
opts.Default = 'Volver';

while isempty(respuesta)
    respuesta = questdlg('\fontsize{10} ¿Quieres entrenar la red neuronal o regresar a la app?',...
                         'Entrenar CNN', 'Entrenar', 'Volver', opts);
    if strcmpi(respuesta, 'Entrenar')
        disp('Comienza el algoritmo de entrenamiento')
        pool = parpool; % Habilita uso de la GPU para cómputo paralelo
        for i = 1:size(PCNNin, 1)
            trainedNet = ClasificadorCNN(PCNNin{i,:}, records);
            confusionM(NumCNN, trainedNet, PCNNin, CNNin, organized)
        end
        delete(pool)
        save(fullfile(processdir, 'Trained.mat'), 'PCNNin',...
                      'trained', 'NumCNN');
        adquisicion(trainedNet, NumCNN)
    elseif strcmpi(respuesta, 'Volver')
        save(fullfile(processdir, 'Processed.mat'), 'PCNNin', 'records');
    end 
end

%% Definición de funciones

function ads = audiorganize(records, palabras, includeFraction)

    isPalabra = ismember(records.Labels, palabras);
    isUnknown = ~ismember(records.Labels,[palabras, "_background_noise_"]);
    mask = rand(numel(records.Labels), 1) < includeFraction;
    isUnknown = isUnknown & mask;
    records.Labels(isUnknown) = categorical("unknown");
    ads = subset(records, isPalabra|isUnknown);
    ads.Labels = removecats(ads.Labels);
    countEachLabel(ads)
    
end
function [adsTrain,adsValidation,adsTest] = divideData(ads)
% Obtiene subconjuntos de validación y test restringidos a la primera
% grabación por persona de cada palabra; selecciona el resto para
% entrenamiento
    Idx = endsWith(ads.Files, '1.wav');
    l = length(ads.Files(Idx));
    nval = floor(l*0.2); % 20 de las muestras únicas
    ntest = floor(l*0.1); % 10 de las muestras únicas
    first = subset(ads, Idx);
    shuffled = shuffle(first);
    adsValidation = subset(shuffled, (1:nval));
    adsTest = subset(shuffled, (nval+1:nval+ntest));
    adsPreTrain = subset(shuffled, (nval+ntest+1:l));
    sdata = subset(ads, ~Idx);
    adsTrain = audioDatastore(cat(1, adsPreTrain.Files, sdata.Files));
    adsTrain.Labels = cat(1, adsPreTrain.Labels, sdata.Labels);
    
end
function Processed = Process(records, s)

    numFiles = length(records.Files);
    Processed = zeros([s.numBands, s.numHops, numFiles]);
    for i = 1:numFiles
        x = read(records);
        filtered = filter(s.filtro, x);
        dsampled = downsample(filtered, 2);
        framed = dsampled(s.Idx).*s.pWindow;a
        Sxx = abs(fft(framed, s.nfft));
        Processed(:,:,i) = s.nFilterBank'*Sxx + sqrt(eps('double'));
        if mod(i,100) == 0
            disp("Procesados " + i + " archivos de " + numFiles)
        end
    end
    Processed = log10(Processed + eps);
    
end
function layers = CNNArchitecture(YTrain)

classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
     
    maxPooling2dLayer([1 13])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];
end
function trainedNet = ClasificadorCNN(Processed, ads, NumCNN)

    augmenter = imageDataAugmenter('RandXTranslation',[-10 10], ...
                                   'RandXScale',[0.8 1.2], ...
                                   'FillValue',log10(eps));
    layers = cell(NumCNN);
    options = cell(NumCNN);
    augT = cell(NumCNN);
    trainedNet = cell(NumCNN);
    
    for i = 1:NumCNN
        sz = size(Processed{i,1});
        specSize = sz(1:2);
        augT{i} = augmentedImageDatastore(specSize, Processed{i,1},...
                                          ads{i,1}.Labels,...
                                          'DataAugmentation', augmenter);
        layers{i} = CNNArchitecture(ads{i,1}.Labels);
        options{i} = CNNOptions(Processed{i,2}, ads{i,1:2}.Labels);
    end
    
    parpool
    for i = 1:NumCNN
        trainedNet{i} = trainNetwork(augT{i},layers{i},options{i});
    end
    
end
function options = CNNOptions(XValidation, YValidation)

    miniBatchSize = 128;
    validationFrequency = floor(numel(YValidation{1}.Labels)/...
                                miniBatchSize);
    options = trainingOptions('adam', ...
        'ExecutionEnvironment', 'gpu',...
        'InitialLearnRate', 3e-4, ...
        'MaxEpochs', 25, ...
        'MiniBatchSize', miniBatchSize, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'training-progress', ...
        'Verbose', false, ...
        'ValidationData', {XValidation, YValidation{2}.Labels}, ...
        'ValidationFrequency', validationFrequency, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 20);
    
end
function confusionM(NumCNN, trainedNet, PCNNin, CNNin, organized)

    formatSpec = 'Matriz de confusión con datos de validación para la red %d';
    for i = 1:NumCNN
        YValPred = classify(trainedNet, PCNNin{i,2});
        validationError = mean(YValPred ~= CNNin{i,2}.Labels);
        YTrainPred = classify(trainedNet,PCNNin{i,1});
        trainError = mean(YTrainPred ~= CNNin{i,1}.Labels);
        disp("Error en conjunto de entrenamiento: " + trainError*100 + "%")
        disp("Error en conjunto de validación: " + validationError*100 + "%")
        figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
        cm = confusionchart(CNNin{i,2}.Labels, YValPred);
        cm.Title = sprintf(formatSpec, i); 
        cm.ColumnSummary = 'column-normalized';
        cm.RowSummary = 'row-normalized';
        sortClasses(cm, [organized{i},"unknown","background"])
    end
end
