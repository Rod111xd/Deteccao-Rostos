package run;

import org.opencv.core.Core;
import deteccao.Deteccao;

public class OpenCVTest {
    public static void main(String[] args) {
        // Carregar o OpenCV
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        // Executar o programa
        new Deteccao().run();
    }
}