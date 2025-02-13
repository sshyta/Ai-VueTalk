import tf, { norm, Tensor } from '@tensorflow/tfjs';

// Данные
import { TRAINING_DATA } from './data/real-estate-data.js';

const INPUT_TENSOR = tf.tensor2d(TRAINING_DATA.inputs);
const OUTPUT_TENSOR = tf.tensor1d(TRAINING_DATA.outputs);

// Нормализация данных
const MIN_INPUT_VALUE = tf.min(INPUT_TENSOR, 0);
const MAX_INPUT_VALUE = tf.max(INPUT_TENSOR, 0);
const normalize = (tensor) => {
    const result = tf.tidy(() => {
        // Создание substract rensor и range size
        const SUBSTRACT_TENSOR = tf.sub(tensor, MIN_INPUT_VALUE);
        const RANGE_SIZE = tf.sub(MAX_INPUT_VALUE, MIN_INPUT_VALUE);
        
        // Нормализация тензора
        const TENSOR = tf.div(SUBSTRACT_TENSOR, RANGE_SIZE);
        return TENSOR;
    })
    return result;
};

const NORMALIZED_INPUT_TENSOR = normalize(INPUT_TENSOR);

// Cоздание модели
let model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 1 }));

// Обучение модели 
const train = async () => {
    // Компиляция модели
    // Указание оптимизатора и функция потерь
    model.compile({
        optimizer: tf.train.sgd(0.01), // learning rate = 0.01
        loss: 'meanSquaredError',
    });

    // Обучение
    await model.fit(NORMALIZED_INPUT_TENSOR, OUTPUT_TENSOR, {
        batchSize: 64,
        epochs: 10,
        shuffle: true,
    });
    // Удаление промежуточных тензоров из памяти
    NORMALIZED_INPUT_TENSOR.dispose();
    OUTPUT_TENSOR.dispose();
};
await train();

// Тестирование модели
const tryToPredict = (array) => {
    tf.tidy (() => {
        // Нормализация входящих данных 
        let input = normalize(tf.tensor2d(array));
        // Предсказание значений
        let output = model.predict(input);
        
        //console.log("Предполагаемая цена: [3056, 3], [1034, 2], [950, 1]: ")
        output.print();
    })
}
tryToPredict([
  [3056, 3], [1034, 2], [950, 1]  
]);

// Удаление оставшихся тензоров и модели 
MIN_INPUT_VALUE.dispose();
MAX_INPUT_VALUE.dispose();
model.dispose();