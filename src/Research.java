import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
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
public class Research {
    private static final Random RANDOM = new Random();
    
    protected int N = 70 + 1;
    protected int M = 10;
    
    protected List<double[][]> trainingSet, validationSet;
    
    public Research(){}
    
    protected void getSets(File folder, int n) throws IOException{
        trainingSet = new ArrayList<double[][]>();
        validationSet = new ArrayList<double[][]>();
        
        int n0 = n/10;
        for(int i=0;i<10;i++)
            getSets(folder,i,n0);
    }
    
    private void getSets(File folder, int i, int n) throws IOException{
        File file;
        File [] files  = folder.listFiles();
        int k=0;
        int j=0;
        while(k<n && j<files.length) {
            file = files[j];
            j++;
            if(file.getName().charAt(0) - 48 == i){
                k++;
                trainingSet.add(getPattern(file));
            }
        }
        
        while(j<files.length){
            file = files[j];
            j++;
            if(file.getName().charAt(0) - 48 == i){
                k++;
                validationSet.add(getPattern(file));
            }
        }
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
//                System.out.println(image.getRGB(j, i));
                if(image.getRGB(j, i)==-1)
                    pattern[0][k] = 1;
                
                k++;
            }
        
        int i = file.getName().charAt(0) - 48;
        
        pattern[1][i] = 1;
        
        return pattern;
    }
    
    protected NeuralNetwork network(int L, double interval){
        double [][] w_h = new double [L][N];
        double [][] w_o = new double [M][L];
        DoubleUnaryOperator [] f_o = new DoubleUnaryOperator[M];
        DoubleUnaryOperator [] f_h = new DoubleUnaryOperator[L];        
        
        double b = interval/2;
        double a = -b;
        
        DoubleUnaryOperator sigmoid = x -> 1.0D/(1+Math.exp(-x));
        DoubleUnaryOperator ones = x -> 1.0D;
        for(int k=0;k<M;k++){
            f_o[k] = sigmoid;
            for(int j=0;j<L;j++){
                w_o[k][j] = RANDOM.nextDouble();
                w_o[k][j] = a + (b-a)*w_o[k][j];
                
            }
        }
        
        for(int j=0;j<L;j++){
            f_h[j] = sigmoid;
            for(int i=0;i<N;i++){
                w_h[j][i] = RANDOM.nextDouble();
                w_h[j][i] = a + (b-a)*w_h[j][i];
            }
        }
        
        f_h[0] = ones;
        
        return new NeuralNetwork(w_h, w_o, f_h, f_o);
    }
    
    protected double error(NeuralNetwork network, Collection<double[][]> validation_set){
        return network.error(validation_set);
    }
    
    public NeuralNetwork learn(NeuralNetwork network, File folder, int n, double nabla, double alpha) throws IOException{
        this.getSets(folder, n);
        System.out.println(network.learn(trainingSet, nabla, alpha));
        return network;
    }
    
    protected double avg(double [] xs){
        double sum = 0;
        for(double x: xs)
            sum += x;
        
        return  sum/xs.length;
    }
}
