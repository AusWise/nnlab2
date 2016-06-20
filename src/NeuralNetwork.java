import java.io.Serializable;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author auswise
 */
public class NeuralNetwork implements Serializable{
    private static final Random RANDOM = new Random();
    
    double [][] w_h, w_o;
    double [] d_h, d_o, d_h_old, d_o_old;
    int N, L, M;
    DoubleUnaryOperator [] f_h, f_o;
    
    public NeuralNetwork(int N, int L, int M, double interval){
        this.N = N;
        this.L = L;
        this.M = M;
   
        w_h = new double [L][N];
        w_o = new double [M][L];
        
        f_o = new DoubleUnaryOperator[M];
        f_h = new DoubleUnaryOperator[L];  
        
        this.d_h = new double [L];
        this.d_o = new double [M];
        this.d_h_old = new double [L];
        this.d_o_old = new double [M];
        
        DoubleUnaryOperator sigmoid = x -> 1.0D/(1+Math.exp(-x));
        DoubleUnaryOperator ones = x -> 1.0D;
        
        this.resetWeights(interval);
        
        for(int k=0;k<M;k++)
            f_o[k] = sigmoid;
            
        for(int j=0;j<L;j++)
            f_h[j] = sigmoid;
            
        f_h[0] = ones;
    }
    
    public NeuralNetwork(double [][] w_h, double [][] w_o, DoubleUnaryOperator [] f_h, DoubleUnaryOperator [] f_o){
        N = w_h[0].length;
        L = w_h.length;
        M = w_o.length;
        this.w_h = w_h;
        this.w_o = w_o;
        this.f_h = f_h;
        this.f_o = f_o;
        this.d_h = new double [L];
        this.d_o = new double [M];
        this.d_h_old = new double [L];
        this.d_o_old = new double [M];
    }
    
    public double [] forward_propagation(double [] x){
        double [] i = forward_propagation_hidden(x);
        double [] o = forward_propagation_out(i);
        return o;
    }
    
    private double [] forward_propagation_out(double [] i){
        double [] o = new double[M];
        DoubleUnaryOperator f;
        double net;
        for(int k=0;k<M;k++){
            net = dot(w_o[k], i);
            f = f_o[k];
            o[k] = f.applyAsDouble(net);
        }
        
        return o;
    }
    
    private double [] forward_propagation_hidden(double [] x){
        double [] i = new double [L];
        DoubleUnaryOperator f;
        double net;
        for(int j=0;j<L;j++){
            net = dot(w_h[j], x);
            f = f_h[j];
            i[j] = f.applyAsDouble(net);
        }
        
        return i;
    }
    
    private double dot(double [] x, double [] y){
        if(x.length != y.length)
            throw new RuntimeException();
        
        int n = x.length;
        double dot = 0;
        for(int i=0;i<n;i++)
            dot += x[i]*y[i];
        
        return dot;
    }
   
    public int learn(List<double [][]> training_set, double nabla, double alpha){
        int T = 1000;
        double eps = 0.01;
        
        double error;
        boolean success;
        for(int t=0;t<T;t++){
            success = true;
            for(int i=0;i<training_set.size();i++)
                this.backward_propagation(training_set.get(i), nabla, alpha);
            
            error = this.error(training_set);
//            System.out.println(String.format("%.10f", error));
            if(error<=eps)
                return t;
            
            Collections.shuffle(training_set);
        }
        
        return T;
    }
    
    private void backward_propagation(double [][] pattern, double nabla, double alpha){
       double [] x = pattern[0];
       double [] y = pattern[1];
       
       double [] i = forward_propagation_hidden(x);
       double [] o = forward_propagation_out(i);
      
       d_h_old = d_h;
       d_o_old = d_o;
       
       this.d_h = new double [L];
       this.d_o = new double [M];
       
       DoubleUnaryOperator f;
       double net;
       for(int k=0;k<M;k++){
           net = dot(w_o[k], i);
           f = f_o[k];
           d_o[k] = (y[k]-o[k]) * derivative(f, net);
       }
       
       for(int j=0;j<L;j++){
           net = dot(w_h[j], x);
           f = f_h[j];
           
           for(int k=0;k<M;k++){
               d_h[j] += d_o[k]*w_o[k][j];
           }
           
           d_h[j] *= derivative(f,net);
       }
       
       for(int k=0;k<M;k++)
           for(int j=0;j<L;j++)
               w_o[k][j] += nabla * (d_o[k] + alpha*d_o_old[k]) * i[j];
       
       for(int j=0;j<L;j++)
           for(int ii=0;ii<N;ii++)
               w_h[j][ii] += nabla * (d_h[j] + alpha*d_h_old[j]) * x[ii];
   }
   
   private double derivative(DoubleUnaryOperator f, double x){
       double h= 0.01;
       double result = (f.applyAsDouble(x+h) - f.applyAsDouble(x))/h;
//       System.out.println(result);
       return result;
   }
   
   public double error(double [][] pattern){
       double [] x = pattern[0];
       double [] y = pattern[1];
       
       double [] o = this.forward_propagation(x);
       double d;
       double sum = 0;
       for(int k=0;k<M;k++){
           d=o[k]-y[k];
           sum += d*d;
       }
       
       return 1.0D/2 * sum;
   }
   
   public double error(Collection<double [][]> patterns){
       double sum = 0;
       double [] o;
       double ok;
       double yk;
       for(double [][] pattern : patterns){
           o = this.forward_propagation(pattern[0]);
           yk = argmax(pattern[1]);
           ok = argmax(o);
           sum += ok!=yk ? 1 : 0;
       }
       
       return sum/patterns.size();
   }
   
   public void resetWeights(double interval){
       double b = interval/2;
       double a = -b;
       
       for(int k=0;k<M;k++)
            for(int j=0;j<L;j++){
                w_o[k][j] = RANDOM.nextDouble();
                w_o[k][j] = a + (b-a)*w_o[k][j];
            }
        
        
        for(int j=0;j<L;j++)
            for(int i=0;i<N;i++){
                w_h[j][i] = RANDOM.nextDouble();
                w_h[j][i] = a + (b-a)*w_h[j][i];
            }
    }
   
   private int argmax(double [] a){
        int argmax = 0;
        for(int i=0;i<a.length;i++)
            if(a[argmax]<a[i])
                argmax=i;
        
        return argmax;
    }
}
