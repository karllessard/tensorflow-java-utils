package com.kubx.tensorflow.data;

import java.nio.DoubleBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class DoubleValue extends Value<DoubleValue, DoubleBuffer> {
 
  public static DoubleValue of(Operand<Double> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static DoubleValue of(Output<Double> output) {
    return of((Tensor<Double>)output.tensor());
  }

  public static DoubleValue of(Tensor<Double> t) {
    return new DoubleValue(t.buffer().asDoubleBuffer(), 0, toIndices(t.shape()));
  }
  
  public double scalar() {
    checkScalar();
    return buffer.get(position);
  }
  
  public void scalar(double scalar) {
    checkScalar();
    buffer.put(position, scalar);
  }
  
  @Override
  protected DoubleValue newValue(DoubleBuffer buffer, int position, List<Index> indices) {
    return new DoubleValue(buffer, position, indices);
  }
  
  @Override
  protected DoubleBuffer sliceBuffer(DoubleBuffer buffer) {
    return buffer.slice();
  }
  
  private DoubleValue(DoubleBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
