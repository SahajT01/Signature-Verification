import React, { useState } from "react";
import { PhotoIcon } from "@heroicons/react/24/solid";

const Form = () => {
    const [genuineImagePreview, setGenuineImagePreview] = useState(null);
    const [genuineImageFile, setGenuineImageFile] = useState(null); // State to hold the file
    const [forgedImagePreview, setForgedImagePreview] = useState(null);
    const [forgedImageFile, setForgedImageFile] = useState(null); // State to hold the file
    const [isLoading, setIsLoading] = useState(false);

    const handleGenuineFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setGenuineImagePreview(URL.createObjectURL(file));
            setGenuineImageFile(file); // Set the file
        }
    };

    const handleForgedFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setForgedImagePreview(URL.createObjectURL(file));
            setForgedImageFile(file); // Set the file
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setIsLoading(true);

        // Construct FormData
        const formData = new FormData();
        // Append files here, ensure you have references to the file inputs
        // For example, formData.append('genuineImage', genuineImage);
        // formData.append('forgedImage', forgedImage);

        // Submit formData to your backend
        console.log("Submitting form...");

        // Reset after submission for demonstration
        setIsLoading(false);
        setGenuineImagePreview(null);
        setForgedImagePreview(null);
        // Also reset the files
        setGenuineImageFile(null);
        setForgedImageFile(null);
    };

    const resetGenuineImage = () => {
        setGenuineImagePreview(null);
        setGenuineImageFile(null);
    };

    const resetForgedImage = () => {
        setForgedImagePreview(null);
        setForgedImageFile(null);
    };

    return (
        <div className="mx-auto max-w-4xl p-8">
            <form onSubmit={handleSubmit} className="space-y-8">
                {/* Genuine Signature Upload */}
                <div>
                    <label className="block text-sm font-medium leading-6 text-gray-900">
                        Genuine Signature
                    </label>
                    <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 p-6">
                        {genuineImagePreview ? (
                            <div>
                                <img
                                    src={genuineImagePreview}
                                    alt="Genuine Signature Preview"
                                    className="max-h-60"
                                />
                                <button
                                    type="button"
                                    onClick={resetGenuineImage}
                                    className="text-center text-sm relative cursor-pointer rounded-md bg-white text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-600 focus-within:ring-offset-2 hover:text-indigo-500"
                                >
                                    Change Image
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-1 text-center">
                                <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                                <div className="flex text-sm text-gray-600">
                                    <label
                                        htmlFor="genuine-signature"
                                        className="relative cursor-pointer rounded-md bg-white text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2 hover:text-indigo-500"
                                    >
                                        <span>Upload a file</span>
                                        <input
                                            id="genuine-signature"
                                            name="genuineSignature"
                                            type="file"
                                            className="sr-only"
                                            onChange={handleGenuineFileChange}
                                        />
                                    </label>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Forged Signature Upload */}
                <div>
                    <label className="block text-sm font-medium leading-6 text-gray-900">
                        Forged Signature
                    </label>
                    <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 p-6">
                        {forgedImagePreview ? (
                            <div>
                                <img
                                    src={forgedImagePreview}
                                    alt="Forged Signature Preview"
                                    className="max-h-60"
                                />
                                <button
                                    type="button"
                                    onClick={resetForgedImage}
                                    className="text-center text-sm relative cursor-pointer rounded-md bg-white text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-600 focus-within:ring-offset-2 hover:text-indigo-500"
                                >
                                    Change Image
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-1 text-center">
                                <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                                <div className="flex text-sm text-gray-600">
                                    <label
                                        htmlFor="forged-signature"
                                        className="relative cursor-pointer rounded-md bg-white text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2 hover:text-indigo-500"
                                    >
                                        <span>Upload a file</span>
                                        <input
                                            id="forged-signature"
                                            name="forgedSignature"
                                            type="file"
                                            className="sr-only"
                                            onChange={handleForgedFileChange}
                                        />
                                    </label>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Submit Button */}
                <div>
                    <button
                        type="submit"
                        className="text-sm font-semibold leading-6 text-white bg-indigo-600 border border-transparent rounded-md shadow-sm px-4 py-2 w-full transition duration-150 ease-in-out hover:bg-indigo-500 focus:outline-none focus:border-indigo-700 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                    >
                        {isLoading ? "Processing..." : "Submit"}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default Form;
