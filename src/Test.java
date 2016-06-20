import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.imageio.ImageIO;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author auswise
 */
public class Test {
    
    static int n=0;
    static double sum = 0;
    
    public static void main(String [] args) throws IOException{
        
//        Research r = new Research();
//        NeuralNetwork network = new DigitRecognizer(100, 1.0);
//        network =  r.learn(network, new File("/home/auswise/Documents/NetBeansProjects/nnlab2/train_val_set"), 400 ,0.5, 0);
//        
        Iterable<double [][]> testSet0 = getTestSet(new File("/home/auswise/Documents/NetBeansProjects/nnlab2/test_sets/0"));
        Iterable<double [][]> testSet1 = getTestSet(new File("/home/auswise/Documents/NetBeansProjects/nnlab2/test_sets/1"));
        Iterable<double [][]> testSet2 = getTestSet(new File("/home/auswise/Documents/NetBeansProjects/nnlab2/test_sets/2"));

        String folder = "/home/auswise/Documents/NetBeansProjects/nnlab2/train_val_set";
//        int L = 100;
//        double nabla = 0.5;
//        double alpha = 0.0;
//        int n = 400;
//        double interval = 1.0;
//        
//        int [] Ls = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
//        Research_L rl = new Research_L(folder, nabla, alpha, n, interval);
//        rl.research(Ls);
//        System.out.println();
//        
//        double [] nablas = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
//        Research_nabla rna = new Research_nabla(folder, L, alpha, n, interval);
//        rna.research(nablas);
//        System.out.println();
//        
//        double [] alphas = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
//        Research_alpha ra = new Research_alpha(folder, L, nabla, n, interval);
//        ra.research(alphas);
//        System.out.println();
//        
//        int [] ns = {100, 200, 300, 400, 500, 600};
//        Research_n rn = new Research_n(folder, L, nabla,alpha, interval);
//        rn.research(ns);
//        System.out.println();
//        
//        double [] intervals = {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0};
//        Research_interval ri = new Research_interval(folder, L, nabla, alpha, n);
//        ri.research(intervals);
//        System.out.println();
//        
        int L = 100;
        double nabla = 0.1;
        double alpha = 0.7;
        int n = 400;
        double interval = 1.0;
//        
        Research r = new Research();
        NeuralNetwork network = new DigitRecognizer(L, interval);
        network =  r.learn(network, new File("/home/auswise/Documents/NetBeansProjects/nnlab2/train_val_set"), n, nabla, alpha);
        
        test(network, testSet0);
        test(network, testSet1);
        test(network, testSet2);
    }
    
    public static void test(NeuralNetwork network, Iterable<double [][]> testSet){
        int a,b;
        for(double[][] pattern : testSet){
            a = argmax(pattern[1]);
            b = argmax(network.forward_propagation(pattern[0]));
            n++;
            sum += (a!=b) ? 1 : 0;
            System.out.println(a + " " + b);
        }
        System.out.println();
        System.out.println(sum/n);
        
    }
    
    private static int argmax(double [] a){
        int argmax = 0;
        for(int i=0;i<a.length;i++)
            if(a[argmax]<a[i])
                argmax=i;
        
        return argmax;
    }
    
    private static Collection<double [][]> getTestSet(File folder) throws IOException{
        List<double [][]> testSet = new ArrayList<double [][]>();
        
        File [] files  = folder.listFiles();
        
        for(File file : files)
            testSet.add(getPattern(file));
        
        return testSet;
    }
    
    private static double [][] getPattern(File file) throws IOException{
        BufferedImage image = ImageIO.read(file);
        
        double pattern[][] = new double [2][];
        pattern[0] = new double[70+1];
        pattern[1] = new double[10];
        int k=0;
        pattern[0][k] = 1;
        
        for(int i=0;i<image.getHeight();i++)
            for(int j=0;j<image.getWidth();j++){
                if(image.getRGB(j, i)==-1)
                    pattern[0][k] = 1;
                
                k++;
            }
        
        int i = file.getName().charAt(0) - 48;
        
        pattern[1][i] = 1;
        
        return pattern;
    }
}