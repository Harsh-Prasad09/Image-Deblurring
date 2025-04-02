import React from "react";
import sharpImg from "../assets/sharp.png";
import blurImg from "../assets/blur.png";

const HeroSection = () => {
  return (
    <section className="py-16 bg-gradient-to-b from-[#95FFFE] to-[#C6EEF6] pb-40">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex flex-col md:flex-row items-center ml-4">
          
          {/* Left: Text Section - More dominant with larger text */}
          <div className="md:w-1/2 lg:w-7/12 mb-8 md:mb-0 md:pr-8">
            <h1 className="text-4xl lg:text-6xl font-bold text-[#022249] mb-6">
              Transform blurry images into crystal clear photos
            </h1>
            <p className="text-xl text-gray-700 mb-8 leading-relaxed">
              Enhance your photos instantly using advanced machine learning.
            </p>
            <a
              href="#try-it"
              className="bg-[#F07C41] text-white px-8 py-4 rounded-lg font-semibold hover:bg-orange-500 transition shadow-lg hover:shadow-xl text-lg"
            >
              Try It Now
            </a>
          </div>

          {/* Right: Side-by-Side Image Comparison - Shifted right */}
          <div className="md:w-1/2 lg:w-6/12 md:pl-4 ml-auto">
            <div className="bg-gradient-to-br from-[#022249] to-[#094681] p-6 rounded-xl shadow-xl max-w-2xl ml-auto border border-[#55BDC9]/30">
              <div className="grid grid-cols-2 gap-4">
                {/* Blurry Image */}
                <div className="bg-white relative overflow-hidden rounded-lg shadow-md h-full">
                  <div className="absolute top-0 left-0 w-full bg-gradient-to-r from-[#F07C41] to-[#f3945f] h-2"></div>
                  <div className="absolute top-4 left-4 bg-[#F07C41] text-white px-4 py-1 rounded-full text-sm font-medium shadow-md z-10">
                    Before
                  </div>
                  <div className="h-60 md:h-72 lg:h-80 overflow-hidden">
                    <img 
                      src={blurImg} 
                      alt="Blurry" 
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
                
                {/* Sharp Image */}
                <div className="bg-white relative overflow-hidden rounded-lg shadow-md h-full">
                  <div className="absolute top-0 left-0 w-full bg-gradient-to-r from-[#55BDC9] to-[#95FFFE] h-2"></div>
                  <div className="absolute top-4 right-4 bg-[#55BDC9] text-white px-4 py-1 rounded-full text-sm font-medium shadow-md z-10">
                    After
                  </div>
                  <div className="h-60 md:h-72 lg:h-80 overflow-hidden">
                    <img 
                      src={sharpImg} 
                      alt="Sharp" 
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>
              </div>
              
              <div className="mt-4 bg-gradient-to-r from-[#95FFFE]/20 to-[#C6EEF6]/20 p-3 rounded-lg backdrop-blur-sm">
                <p className="text-center text-[#95FFFE] font-medium">
                  Observe the difference we make
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;