import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class ImageClassification {
    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        Classifications classifications = ImageClassification.predict();
        System.out.println(classifications);
    }

    static class MyTranslator implements Translator<Image, Classifications> {
        private List<String> classes;
        private SynsetLoader synsetLoader = new SynsetLoader("synset.txt");

        static final class SynsetLoader {
            private String synsetFileName;

            public SynsetLoader(String synsetFileName) {
                this.synsetFileName = synsetFileName;
            }

            public List<String> load(Model model) throws IOException {
                return model.getArtifact(synsetFileName, Utils::readLines);
            }
        }

        @Override
        public void prepare(NDManager manager, Model model) throws IOException {
            if (classes == null) {
                classes = synsetLoader.load(model);
            }
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilitiesNd = list.singletonOrThrow();
            // TODO Implement softmax here
            return new Classifications(classes, probabilitiesNd);
        }

        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            array = NDImageUtils.resize(array, 256, 256);
            array = NDImageUtils.centerCrop(array, 224, 224);

            // TODO Implement toTensor() here
            // remove the reshape here
            array = array.reshape(new Shape(3, 224, 224));

            array = NDImageUtils.normalize(array, new float[] {0.485f, 0.456f, 0.406f}, new float[] {0.229f, 0.224f, 0.225f});
            return new NDList(array);
        }
    }

    public static Classifications predict() throws IOException, ModelException, TranslateException {
        Image img = ImageFactory.getInstance().fromUrl("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg");

        // TODO implement model loading using criteria
        Criteria<Image, Classifications> criteria = ...

        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, Classifications> predictor = model.newPredictor()) {
                return predictor.predict(img);
            }
        }
    }
}
