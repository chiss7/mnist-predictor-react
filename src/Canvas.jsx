import { useEffect, useState } from "react";
import { fabric } from "fabric";

const Canvas = (props) => {
  const { id, height, width, setClear, className, onCanvasReady } = props;
  const [canvas, setCanvas] = useState(null);

  useEffect(() => {
    const newCanvas = new fabric.Canvas(id, {
      isDrawingMode: true,
    });

    newCanvas.freeDrawingBrush.width = 10;
    setCanvas(newCanvas);

    if (onCanvasReady) {
      onCanvasReady(newCanvas);
    }

    return () => {
      newCanvas.dispose();
    };
  }, [id, onCanvasReady]);

  const handleClearCanvas = () => {
    if (canvas) {
      canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      canvas.clear();
    }
  };

  return (
    <div>
      {setClear && <button onClick={handleClearCanvas}>Clear</button>}
      <canvas id={id} height={height} width={width} className={className} />
    </div>
  );
};

export default Canvas;