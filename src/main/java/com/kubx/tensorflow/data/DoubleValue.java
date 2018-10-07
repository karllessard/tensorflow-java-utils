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
    return of(t.buffer().asDoubleBuffer(), t.shape());
  }

  public static DoubleValue of(DoubleBuffer buffer, long[] shape) {
    return new DoubleValue(buffer, 0, toIndices(shape));
  }
  
  public double scalar() {
    if (indices.size() > 0) {
      throw new IllegalArgumentException("Cannot convert value of " + indices.size() + " dimensions to scalar");
    }
    return buffer.get(0);
  }
  
  @Override
  protected DoubleValue newValue(DoubleBuffer buffer, int position, List<Index> indices) {
    return new DoubleValue(buffer, position, indices);
  }
  
  @Override
  protected DoubleBuffer slice() {
    return buffer.slice();
  }
  
  private DoubleValue(DoubleBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
