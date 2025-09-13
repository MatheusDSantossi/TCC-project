import { Suspense, useEffect, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Preload, useGLTF } from "@react-three/drei";

import CanvasLoader from "../Loader";
// import scene from '../../../public/scene.gltf/'

const Rocket = ({ isMobile }) => {
  const ref = useRef(null);

  const rocket = useGLTF("./rocket/scene.gltf");

  useFrame((state, delta) => {
    if (ref.current) {
      ref.current.rotation.x += 0.5 * delta;
      // ref.current.rotation.z += 0.001 * delta;

      if (ref.current.position.x > 10) {
        ref.current.position.x = -10;
      }
    }
  });

  return (
    <mesh ref={ref}>
      <hemisphereLight intensity={0.99} groundColor="black" />
      <primitive
        object={rocket.scene}
        scale={isMobile ? 0.01 : 0.01}
        position={isMobile ? [0, -7, 0] : [0, -10, 0]}
        rotation={[-0.01, 1, 0.1]}
      />
    </mesh>
  );
};

const RocketCanvas = () => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    // Add a listener for changes to the screen size
    const mediaQuery = window.matchMedia("(max-width: 500px)");

    // Set the initial value of the `isMobile` state
    const handleMediaQueryChange = (event) => {
      setIsMobile(event.matches);
    };

    // Add the callback function as a listener for changes to the media query
    mediaQuery.addEventListener("change", handleMediaQueryChange);

    // Remove the listener when the component is unmounted
    return () => {
      mediaQuery.removeEventListener("change", handleMediaQueryChange);
    };
  }, []);

  return (
    <div className="fixed top-0 left-0 w-full h-full z-10">
      <Canvas
        frameloop="always"
        shadows
        dpr={[1, 2]}
        camera={{ position: [20, 3, 5], fov: 25 }}
        gl={{ preserveDrawingBuffer: true }}
      >
        <Suspense fallback={<CanvasLoader />}>
          <OrbitControls
            enableZoom={false}
            maxPolarAngle={Math.PI / 2}
            minPolarAngle={Math.PI / 2}
          />
          <Rocket isMobile={isMobile} />
        </Suspense>
        <Preload all />
      </Canvas>
    </div>
  );
};

export default RocketCanvas;
