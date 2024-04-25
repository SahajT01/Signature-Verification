import React, { useState } from "react";
import { PhotoIcon } from "@heroicons/react/24/solid";

const CreateUserForm = () => {
    const [genuineImagePreview, setGenuineImagePreview] = useState(null);
    const [genuineImageFile, setGenuineImageFile] = useState(null);

    const resetGenuineImage = () => {
        setGenuineImagePreview(null);
        setGenuineImageFile(null);
    };

    const handleGenuineFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setGenuineImagePreview(URL.createObjectURL(file));
            setGenuineImageFile(file); // Set the file
        }
    };

    return (
        <div>
            <div className="flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8">
                <div className="sm:mx-auto sm:w-full sm:max-w-sm">
                    {/* <img
                        className="mx-auto h-10 w-auto"
                        src="https://tailwindui.com/img/logos/mark.svg?color=indigo&shade=600"
                        alt="Your Company"
                    /> */}
                    <h2 className="mt-10 text-center text-2xl font-bold leading-9 tracking-tight text-gray-900">
                        Create a User
                    </h2>
                </div>

                <div className="mt-10 sm:mx-auto sm:w-full sm:max-w-sm">
                    <form className="space-y-6" action="#" method="POST">
                        <div>
                            <label className="block text-sm font-medium leading-6 text-gray-900">
                                Name
                            </label>
                            <div className="mt-2">
                                <input
                                    type="text"
                                    required
                                    className="block w-full rounded-md border-0 p-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                />
                            </div>
                        </div>

                        <div>
                            <label
                                htmlFor="email"
                                className="block text-sm font-medium leading-6 text-gray-900"
                            >
                                Email address
                            </label>
                            <div className="mt-2">
                                <input
                                    type="email"
                                    autoComplete="email"
                                    required
                                    className="block w-full rounded-md border-0 p-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                />
                            </div>
                        </div>

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
                                                    onChange={
                                                        handleGenuineFileChange
                                                    }
                                                />
                                            </label>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div>
                            <button
                                type="submit"
                                className="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                            >
                                Create
                            </button>
                        </div>
                    </form>

                    {/* <p className="mt-10 text-center text-sm text-gray-500">
                        Not a member?{" "}
                        <a
                            href="#"
                            className="font-semibold leading-6 text-indigo-600 hover:text-indigo-500"
                        >
                            Start a 14 day free trial
                        </a>
                    </p> */}
                </div>
            </div>
        </div>
    );
};

export default CreateUserForm;
