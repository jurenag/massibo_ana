import os
from typing import Tuple, Union, Generator
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4 # 595.28 width x 841.89 height, in points
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from PyPDF2 import PdfReader, PdfWriter, PdfMerger

import matplotlib
import matplotlib.figure

import src.utils.htype as htype

class PDFGenerator:

    def __init__(
            self,
            output_filepath: str,
            pagesize: Tuple[float, float] = A4
        ):
        """This class aims to model an open and evolving PDF file, to which
        we can add plots, using the add_plot() method, and text, using the
        add_text() method. The PDF file is managed using the reportlab library.
        Once the PDF is finished, it can be saved with the save() method.

        Parameters
        ----------
        output_filepath: str
            The path to the output PDF file
        pagesize: tuple of two floats
            The size of the pages of the PDF file, in points. The default
            value is A4. Further formats can be found in the
            reportlab.lib.pagesizes module, such as the letter format.
        """
        
        self.__output_filepath = output_filepath
        self.__pagesize = pagesize
        self.__current_page = 1

        self.__canvas = canvas.Canvas(
            output_filepath, 
            pagesize=pagesize
        )

    @property
    def OutputFilepath(self):
        return self.__output_filepath
    
    @property
    def PageSize(self):
        return self.__pagesize
    
    @property
    def CurrentPage(self):
        return self.__current_page
    
    @property
    def Canvas(self):
        return self.__canvas

    def add_plot(
            self,
            figure_to_add: matplotlib.figure.Figure,
            horizontal_pos_frac: float,
            vertical_pos_frac: float,
            plot_width_wrt_page_width: float = .99,
            horizontally_center: bool = False,
        ) -> None:
        
        """This method gets a matplotlib.figure.Figure object,
        figure_to_add, and adds it as a PNG image to the PDF.

        Parameters
        ----------
        figure_to_add: matplotlib.figure.Figure
            The figure that will be added to the PDF as a PNG image.
        horizontal_pos_frac (resp. vertical_pos_frac): float
            It must be a float which is smaller than 1.0. It represents
            the horizontal (resp. vertical) position of the image in the
            page, as a fraction of the page width (resp. height).
            Particularly, it gives the position of the lower left corner
            of the added plot, with respect to the lower left corner of
            the page. I.e. a bigger value of horizontal_pos_frac 
            (resp. vertical_pos_frac) means that the plot is more to the
            right (resp. upper part) in the page. Allowing negative
            values is convenient to create tight layouts with images
            which have big blank frames.
        plot_width_wrt_page_width: float
            It must be a float in the (0., 1.) range. The width of the
            plot is scaled so that the ratio between its final width
            and the page width equals this input. The final height of
            the plot is scaled accordingly to preserve the original
            width-height ratio of the given figure. For example, if
            this input is 0.5, then the width of the added image is
            half the page width.
        horizontally_center: bool
            If True, then the input given to horizontal_pos_frac is
            ignored and the image is centered horizontally in the page.

        Returns
        ----------
        None
        """

        if not horizontally_center:
            if horizontal_pos_frac >= 1.:
                raise ValueError(htype.generate_exception_message(
                        "PDFGenerator.add_plot",
                        1,
                        extra_info="The given horizontal_pos_frac "
                        f"({horizontal_pos_frac}) must be smaller "
                        "than 1.0.",
                    ))
        
        if vertical_pos_frac >= 1.:
            raise ValueError(htype.generate_exception_message(
                    "PDFGenerator.add_plot",
                    2,
                    extra_info="The given vertical_pos_frac "
                    f"({vertical_pos_frac}) must be smaller "
                    "than 1.0.",
                ))
        
        if plot_width_wrt_page_width <= 0. or plot_width_wrt_page_width >= 1.:
            raise ValueError(htype.generate_exception_message(
                    "PDFGenerator.add_plot",
                    3,
                    extra_info="The given plot_width_wrt_page_width "
                    f"({plot_width_wrt_page_width}) must be a float "
                    "in the (0.0, 1.0) range.",
                ))
        
        plot_width_in_points = plot_width_wrt_page_width * self.__pagesize[0]

        if not horizontally_center:
            aux_x = horizontal_pos_frac * self.__pagesize[0]
        else:
            aux_x = (self.__pagesize[0] - plot_width_in_points) / 2

        with tempfile.NamedTemporaryFile(
            suffix=".png", 
            delete=False) as temp_image:

            figure_to_add.savefig(
                temp_image.name, 
                format='png'
            )

            matplotlib.pyplot.close()
            
            self.__canvas.drawImage(
                temp_image.name,
                # x and y are the coordinates, in points,
                # of the lower left corner of the image
                x = aux_x,
                y = vertical_pos_frac * self.__pagesize[1],
                # width is the resulting-image width in points
                # One inch is 72 points
                width=plot_width_in_points,
                preserveAspectRatio=True)
            
    def add_text(
            self,
            text: str,
            horizontal_pos_frac: float,
            vertical_pos_frac: float,
            max_width_frac: float = 0.8,
            font: str = "Helvetica",
            font_size: float = 12.,
            font_color: colors.Color = colors.black,
            horizontally_center: bool = False
        ) -> None:
        """This method adds a block of text to the PDF at a specific position.
        
        Parameters
        ----------
        text: str 
            The text to add
        horizontal_pos_frac (resp. vertical_pos_frac): float
            It must be a float in the [0., 1.) range. It represents the
            horizontal (resp. vertical) position of the text block in the
            page, as a fraction of the page width (resp. height). 
            Particularly, it gives the position of the lower left corner
            of the added text block, with respect to the lower left corner
            of the page. I.e. a bigger value of horizontal_pos_frac 
            (resp. vertical_pos_frac) means that the text block is more
            to the right (resp. upper part) in the page.
        max_width_frac: float 
            It must be a float in the (0., 1.] range. It is the maximum
            width of the text block, as a fraction of the page width.
        font: str 
            Text font
        font_size: float
            It must be a positive float. It is the text size, in points.
        font_color: reportlab.lib.colors.Color
            The color of the added text
        horizontally_center: bool
            If True, then the input given to horizontal_pos_frac is
            ignored and the text block is centered horizontally in the
            page.
        """

        if not horizontally_center:
            if horizontal_pos_frac < 0. or horizontal_pos_frac >= 1.:
                raise ValueError(htype.generate_exception_message(
                        "PDFGenerator.add_text",
                        1,
                        extra_info="The given horizontal_pos_frac "
                        f"({horizontal_pos_frac}) must belong to "
                        "the [0.0, 1.0) range.",
                    ))
        
        if vertical_pos_frac < 0. or vertical_pos_frac >= 1.:
            raise ValueError(htype.generate_exception_message(
                    "PDFGenerator.add_text",
                    2,
                    extra_info="The given vertical_pos_frac "
                    f"({vertical_pos_frac}) must belong to "
                    "the [0.0, 1.0) range.",
                ))
        
        if max_width_frac <= 0. or max_width_frac > 1.:
            raise ValueError(htype.generate_exception_message(
                    "PDFGenerator.add_text",
                    3,
                    extra_info="The given max_width_frac "
                    f"({max_width_frac}) must belong to "
                    "the (0.0, 1.0] range.",
                ))
        
        if font_size <= 0.:
            raise ValueError(htype.generate_exception_message(
                    "PDFGenerator.add_text",
                    4,
                    extra_info="The given font_size "
                    f"({font_size}) must be a positive float.",
                ))

        self.__canvas.setFont(
            font, 
            font_size
        )

        # Compute the text-block width, in points
        max_text_width_in_points = max_width_frac * self.__pagesize[0]

        lines = simpleSplit(
            text,
            font,
            font_size,
            max_text_width_in_points)
        
        if not horizontally_center:
            aux_x = horizontal_pos_frac * self.__pagesize[0]
        else:
            aux_x = (self.__pagesize[0] - max_text_width_in_points) / 2

        self.__canvas.setFillColor(font_color)

        for i, line in enumerate(lines):
            self.__canvas.drawString(
                aux_x,
                (vertical_pos_frac * self.__pagesize[1]) - (i * font_size),
                line
            )

        # Set the default canvas color back to black
        # I think this line is not needed though
        self.__canvas.setFillColor(colors.black)
        

    def close_page_and_start_a_new_one(self) -> None:
        """This method ends the current page, 'sends' it to
        the PDF file, and starts a new one. After calling this
        method, any added content will be placed in a new page
        blank page."""

        self.__canvas.showPage()
        self.__current_page += 1


    def save(self) -> None:
        """This method saves the PDF to the specified output file."""

        self.__canvas.save()

    @staticmethod
    def add_cover(
        path_to_cover_pdf: str, 
        path_to_body_pdf: str, 
        path_to_output_pdf: str
        ) -> None:
        """This method gets the path to two existing PDF files: a cover
        (front page) and a body. It creates a new PDF file that is the
        page-wise concatenation of the cover (actually, the first page
        of the given cover PDF) and the body, in such order. The resulting
        concatenated PDF is saved to the specified output file path.
        Contrary to the rest of the PDFGenerator methods, which use the
        ReportLab library, this method uses the PyPDF2 library.

        Parameters
        ----------
        path_to_cover_pdf: str
            File path to the front-page PDF
        path_to_body_pdf: str 
            File path to the main PDF
        path_to_output_pdf: str
            File path to the output (combined) PDF
        """

        if not os.path.isfile(path_to_cover_pdf) or \
            not path_to_cover_pdf.endswith('.pdf'):

            raise ValueError(htype.generate_exception_message(
                "PDFGenerator.add_cover",
                1,
                extra_info="The given path_to_cover_pdf must point to an existing PDF file."
            ))
        
        if not os.path.isfile(path_to_body_pdf) or \
            not path_to_body_pdf.endswith('.pdf'):

            raise ValueError(htype.generate_exception_message(
                "PDFGenerator.add_cover",
                2,
                extra_info="The given path_to_body_pdf must point to an existing PDF file."
            ))
        
        if not path_to_output_pdf.endswith('.pdf'):

            raise ValueError(htype.generate_exception_message(
                "PDFGenerator.add_cover",
                3,
                extra_info="The given path_to_output_pdf must end with the '.pdf' extension."
            ))

        cover = PdfReader(path_to_cover_pdf)
        body = PdfReader(path_to_body_pdf)

        writer = PdfWriter()

        # Add the cover page first
        writer.add_page(cover.pages[0])

        # Then iteratively add the body pages
        for page in body.pages:
            writer.add_page(page)

        with open(path_to_output_pdf, "wb") as file:
            writer.write(file)

        return
    
    @staticmethod
    def smart_close_page(
            pdf_generator: 'PDFGenerator',
            max_pages_per_pdf_chunk: int,
            pdf_output_filepath_generator: Generator[str, None, None]
    ) -> Tuple[bool, Union[None, 'PDFGenerator']]:
        """The behaviour of this static method depends on the
        current number of pages of the given PDFGenerator object,
        i.e. on its CurrentPage attribute.
        
            - If it is smaller than max_pages_per_pdf_chunk,
        then this static method closes the current page, starts a
        new one, and returns (False, None). 
            
            - If it is bigger or equal to max_pages_per_pdf_chunk,
        then this static method closes the current page, saves the
        PDFGenerator object by calling its 'save' method, and then
        creates a new PDFGenerator by calling 

            PDFGenerator.__init__(
                next(pdf_output_filepath_generator), 
                pdf_generator.PageSize
            )
        
        I.e. the page size of the new PDFGenerator object is the
        same as that of the given PDFGenerator object, and its
        output file path is yielded by the pdf_output_filepath_generator
        generator. This static method then returns 
        (True, new_pdf_generator), where new_pdf_generator is the newly
        created PDFGenerator object.

        Parameters
        ----------
        pdf_generator: PDFGenerator 
            The PDFGenerator object whose current page will be 
            closed
        max_pages_per_pdf_chunk: int 
            It must be positive. If the CurrentPage attribute of
            the given PDFGenerator object is bigger or equal to
            this number, then the PDFGenerator object will be saved
            and a new one will be created and returned. If else,
            the current page is closed and a new one is started.
        pdf_output_filepath_generator: Generator[str, None, None] 
            If a new PDFGenerator object is created, then this
            generator is used to get the file path of the new PDF
            file. 
        
        Returns
        ----------
        output: Tuple[bool, Union[None, PDFGenerator]]
        """

        if max_pages_per_pdf_chunk <= 0:
            raise ValueError(
                htype.generate_exception_message(
                    "PDFGenerator.smart_close_page",
                    1,
                    extra_info="The given max_pages_per_pdf_chunk "
                    f"({max_pages_per_pdf_chunk}) must be positive."
                )
            )
        
        if pdf_generator.CurrentPage < max_pages_per_pdf_chunk:
            pdf_generator.close_page_and_start_a_new_one()
            return False, None
        else:
            pdf_generator.close_page_and_start_a_new_one()
            pdf_generator.save()

            new_pdf_generator = PDFGenerator(
                next(pdf_output_filepath_generator),
                pagesize=pdf_generator.PageSize
            )
            return True, new_pdf_generator

    @staticmethod
    def PDF_chunk_filepath(
            output_folderpath: str,
            output_filename: str
    ) -> Generator[str, None, None]:
        """This static method is a generator that yields the file
        paths of subsequent PDF files, each of which is considered
        to be a chunk of a bigger PDF file, i.e. they might be
        concatenated into a single PDF file considering the order
        yielded by this generator. The file path to the i-th chunk
        is given by:
            
                os.path.join(
                    output_folderpath, 
                    f"chunk_{i}_{output_filename}"
                )

        Parameters
        ----------
        output_folderpath: str 
            The folder where the PDF chunks will be potentially stored
        output_filename: str
            The last part of the PDF file name, common to all the chunks
        """

        n = 0

        while True:
            yield os.path.join(
                output_folderpath,
                f"chunk_{n}_{output_filename}"
            )
            
            n += 1
    
    @staticmethod
    def concatenate_PDFs(
            input_pdf_filepaths: Tuple[str, ...],
            output_pdf_filepath: str
    ) -> None:
        """This static method gets a tuple of file paths to existing
        PDF files, input_pdf_filepaths, and concatenates them into
        a single PDF file, which is saved to the specified output
        file path, output_pdf_filepath. This method uses the PyPDF2
        library.

        Parameters
        ----------
        input_pdf_filepaths: Tuple[str, ...]
            Tuple of file paths to existing PDF files
        output_pdf_filepath: str
            File path to the output (concatenated) PDF file
        """

        if not all(
            map(
                lambda x: os.path.isfile(x) and x.endswith('.pdf'),
                input_pdf_filepaths
            )
        ):
            raise ValueError(htype.generate_exception_message(
                "PDFGenerator.concatenate_PDFs",
                1,
                extra_info="All the given input_pdf_filepaths must point to existing PDF files."
            ))
        
        if not output_pdf_filepath.endswith('.pdf'):
            raise ValueError(htype.generate_exception_message(
                "PDFGenerator.concatenate_PDFs",
                2,
                extra_info="The given output_pdf_filepath must end with the '.pdf' extension."
            ))

        merger = PdfMerger()

        for pdf_filepath in input_pdf_filepaths:
            merger.append(pdf_filepath)

        with open(output_pdf_filepath, "wb") as file:
            merger.write(file)

        return