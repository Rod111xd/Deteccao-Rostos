package deteccao;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Deteccao {
	
	public void detectAndDisplay(Mat frame, CascadeClassifier faceCascade, CascadeClassifier eyesCascade) {
        Mat frameGray = new Mat();
        
        // Escala de cinza
        Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);
        
        // Equaliza��o de histograma
        Imgproc.equalizeHist(frameGray, frameGray);
        
        // Detec��o de rostos
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(frameGray, faces);
        List<Rect> listOfFaces = faces.toList(); // Lista de rostos encontrados
        
        for (Rect face : listOfFaces) {
            Point center = new Point(face.x + face.width / 2, face.y + face.height / 2); // Definir ponto central do rosto
            Imgproc.ellipse(frame, center, new Size(face.width / 2, face.height / 2), 0, 0, 360,
                    new Scalar(255, 0, 255)); // Criar uma circunfer�ncia baseada no tamanho do rosto
            Mat faceROI = frameGray.submat(face); // Submatriz
            
            // Detectar olhos em cada rosto
            MatOfRect eyes = new MatOfRect();
            eyesCascade.detectMultiScale(faceROI, eyes);
            List<Rect> listOfEyes = eyes.toList(); // Lista de olhos encontrados 
            
            for (Rect eye : listOfEyes) {
                Point eyeCenter = new Point(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2); // Ponto central do olho
                int radius = (int) Math.round((eye.width + eye.height) * 0.25); // Raio do c�rculo a ser gerado
                Imgproc.circle(frame, eyeCenter, radius, new Scalar(255, 0, 0), 4); // Criar uma circunfer�ncia para o olho
            }
        }
        // Exibir frame
        HighGui.imshow("Detec��o de rosto e olhos", frame );
    }
	
    public void run() {
    	// Localiza��o dos cascades utilizados na detec��o
        String filenameFaceCascade = "haarcascade_frontalface_alt.xml";
        String filenameEyesCascade = "haarcascade_eye_tree_eyeglasses.xml";

        CascadeClassifier faceCascade = new CascadeClassifier();
        CascadeClassifier eyesCascade = new CascadeClassifier();
        
        // Carregar haarcascade para Rosto
        if (!faceCascade.load(filenameFaceCascade)) {
            System.err.println("Erro ao carregar o face cascade: " + filenameFaceCascade);
            System.exit(0);
        }
        
        // Carregar haarcascade para os olhos
        if (!eyesCascade.load(filenameEyesCascade)) {
            System.err.println("Erro ao carregar o eyes cascade: " + filenameEyesCascade);
            System.exit(0);
        }
        
        // Iniciar captura da Webcam
        VideoCapture capture = new VideoCapture(0);
        
        if (!capture.isOpened()) {
            System.err.println("Erro ao abrir captura de v�deo");
            System.exit(0);
        }
  
        Mat frame = new Mat();
        
        while (capture.read(frame)) {
        	
            if (frame.empty()) {
                System.err.println("Nenhum frame capturado");
                break;
            }
            // Realizar detec��o de rostos e olhos e mostrar nos frames da janela
            detectAndDisplay(frame, faceCascade, eyesCascade);
            if (HighGui.waitKey(10) == 27) {
                break; // Sair
            }
           
        }
       
        System.exit(0);
    }
    
}
