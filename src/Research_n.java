
import java.io.File;
import java.io.IOException;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author auswise
 */
public class Research_n extends Research {
    private double nabla;
    private double alpha; 
    private double interval;
    
    private File folder;
    
    NeuralNetwork network;
    
    public Research_n(String folder, int L, double nabla, double alpha, double interval) {
        this.folder = new File(folder);
        this.nabla = nabla;
        this.alpha = alpha;
        this.interval = interval;
        
        this.network = super.network(L, interval);
    }
    
    public void research(int [] ns) throws IOException{
        for(int n : ns){
            learn(n);
            
        }
    }
    
    public void learn(int n) throws IOException{
        int m = 10;
        double [] ts = new double[m];
        double [] es = new double[m];
        
        getSets(folder, n);
        
        for(int i=0; i<m; i++){
            ts[i] = network.learn(trainingSet, nabla, alpha);
            es[i] = error(network, validationSet);
            network.resetWeights(interval);
        }
        
        System.out.println(n + " " + super.avg(ts) + " " + super.avg(es));
    }
}
