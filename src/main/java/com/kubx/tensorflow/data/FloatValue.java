package com.kubx.tensorflow.data;

import java.nio.FloatBuffer;
import java.util.List;

import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class FloatValue extends Value<FloatValue, FloatBuffer> {
 
  public static FloatValue of(Operand<Float> op) {
    return of(op.asOutput());
  }

  @SuppressWarnings("unchecked")
  public static FloatValue of(Output<Float> output) {
    return of((Tensor<Float>)output.tensor());
  }

  public static FloatValue of(Tensor<Float> t) {
    return new FloatValue(t.buffer().asFloatBuffer(), 0, toIndices(t.shape()));
  }
  
  public float scalar() {
    checkScalar();
    return buffer.get(position);
  }
  
  public void scalar(float scalar) {
    checkScalar();
    buffer.put(position, scalar);
  }
  
  @Override
  protected FloatValue newValue(FloatBuffer buffer, int position, List<Index> indices) {
    return new FloatValue(buffer, position, indices);
  }
  
  @Override
  protected FloatBuffer sliceBuffer(FloatBuffer buffer) {
    return buffer.slice();
  }
  
  private FloatValue(FloatBuffer buffer, int position, List<Index> indices) {
    super(buffer, position, indices);
  }
}
